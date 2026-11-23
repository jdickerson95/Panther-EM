r"""After-CTF noise normalization: every ``(plane, template)`` hypothesis at one noise.

Why
---
Panther normalizes each whitened template projection ``m_n`` to unit variance
*before* the CTF and applies the CTF to the image, one plane per searched defocus
``d``. Under the coloured-noise model the noise variance of hypothesis ``(d, n)`` is
then

.. math:: e_{dn} = \sum_F \mathrm{noise}_{dF} \, P_{nF},

with :math:`P_{nF}` the ring power of template row ``n`` (its error table's
``reconstruction_autocorr``) and :math:`\mathrm{noise}_{dF}` the micrograph's noise
power under ``CTF_d`` (:attr:`~panther_em.search.micrograph.Micrograph.noise`; the
same product the stage-1 threshold test calls ``q_kept``). ``e`` varies with ``n``
(how much of a projection's power falls where the CTF passes it) and with ``d`` (the
CTF's own power). It does not depend on the in-plane angle for an isotropic CTF (a
rotation keeps ring sums); astigmatism is ignored.

The z-score divides every hypothesis by one pooled per-pixel sigma, so hypothesis
``h``'s null z has variance :math:`v_h = e_h / \bar e` instead of 1. A particle's z
moves by :math:`\sqrt{v_h}` (no change on average), but the maximum over millions of
noise hypotheses is drawn preferentially from those with :math:`v_h > 1`: with
:math:`v_h = 1 + \epsilon_h` of relative spread :math:`\sigma` the null tail at
threshold ``t`` is inflated by about :math:`\exp(t^4 \sigma^2 / 8)` and the
null-calibrated cut rises by about :math:`t^3 \sigma^2 / 8`. Leopard-EM and cisTEM
avoid this by normalizing each CTF-filtered projection, i.e. *after* the CTF.

What
----
``normalization: after_ctf`` (:class:`~panther_em.managers.search.SearchConfig`)
multiplies every hypothesis by

.. math:: s_{dn} = (e_{dn} / \bar e)^{-1/2},

the exact per-``(plane, template row)`` standardization (the generalized likelihood
ratio statistic of a template of unknown amplitude), computed per micrograph from
its own noise curves (:func:`normalize`; it sets
:attr:`Micrograph.hypothesis_scale <panther_em.search.micrograph.Micrograph>`). The
mean of ``e`` over the hypotheses is unchanged, so the z-score's scale is too. The
search carries the factor in three places:

* the dense loop's reduction multiplies every hypothesis's correlogram by
  :math:`s_{dn}` (the fused kernel after its in-plane maximum: ``max(s c) = s max(c)``
  and the moment sums take ``s`` and ``s^2``), from a per-loop buffer written per
  pass, so the captured graph and the GEMMs are unchanged;
* a local pass scales its neighborhood spectra and builds its closed-form moment
  forms per plane, :math:`Q_{dk} = \sum_n s_{dn}^2 (\cdot)` (built per micrograph,
  evaluated at the same cost);
* the threshold test multiplies ``q_kept`` and ``q_perp`` by :math:`s_{dn}^2`
  (``lam`` and ``rho`` are ratios).

The variance equalized is that of the correlation of the template's *kept* part
``m~`` in the final stage's feature rectangles (that stage's table), the correlation
the reported z-scores come from; earlier stages use the same factors, so their
threshold tests predict the same scaled correlation.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from panther_em.approximation.errors import PerSpatialFrequencyErrorTable
    from panther_em.search.micrograph import Micrograph


def template_power(table: PerSpatialFrequencyErrorTable) -> torch.Tensor:
    """``(N, F)`` float64 ring power of every row's kept template ``m~``."""
    return table.reconstruction_autocorr.to(torch.float64)


def hypothesis_variance(noise: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
    """``(D, N)`` float64 noise variance ``e`` of every ``(plane, row)``.

    ``noise (D, F)`` the noise power per plane and ring, ``power (N, F)`` the
    templates' ring power; the threshold test's ``q_kept`` at unit local level.
    """
    return noise.to(torch.float64) @ power.to(torch.float64).T


def hypothesis_scale(variance: torch.Tensor) -> torch.Tensor:
    """``(D, N)`` float64 ``s = (e / mean e)^(-1/2)`` of a ``(D, N)`` variance ``e``."""
    e = variance.to(torch.float64)
    if not bool(torch.isfinite(e).all()) or bool((e <= 0).any()):
        raise ValueError("hypothesis variances must be finite and positive")
    return (e / e.mean()).rsqrt()


def relative_sd(values: torch.Tensor) -> float:
    """``sd / mean`` over every element."""
    v = values.to(torch.float64)
    return float(v.std(unbiased=False) / v.mean())


def normalize(
    micrograph: Micrograph, table: PerSpatialFrequencyErrorTable
) -> tuple[Micrograph, dict]:
    """``micrograph`` with every hypothesis scaled to one noise variance, and a report.

    Parameters
    ----------
    micrograph : Micrograph
        Prepared on ``table``'s frequency axis; its noise curves set the factors.
    table : PerSpatialFrequencyErrorTable
        The error table whose kept correlation is equalized (the final stage's).

    Returns
    -------
    tuple[Micrograph, dict]
        The micrograph with :attr:`~Micrograph.hypothesis_scale` set, and a report:
        the relative sd of ``e`` over every ``(plane, row)`` before, and the range and
        relative sd of the factors.
    """
    if not torch.equal(micrograph.frequencies, table.frequencies):
        raise ValueError("the micrograph's curves are on another frequency axis")
    e = hypothesis_variance(micrograph.noise, template_power(table))
    scale = hypothesis_scale(e)
    report = {
        "variance_rel_sd_before": relative_sd(e),
        "scale": {
            "min": float(scale.min()),
            "max": float(scale.max()),
            "rel_sd": relative_sd(scale),
        },
    }
    return replace(micrograph, hypothesis_scale=scale), report
