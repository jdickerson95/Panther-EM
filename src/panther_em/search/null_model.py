r"""Analytic null of a Panther-EM search: its null peak rate without a null search.

A phase-randomised micrograph keeps the micrograph's Fourier amplitudes and draws its
phases at random, so the correlation of the search image with the search templates is
a Gaussian random field over position and hypothesis whose covariance is fixed by the
templates and the image power spectrum. From that covariance Leopard-EM's engine-
agnostic analytic null (:mod:`leopard_em.analysis.analytic_null`) predicts the
expected number of distinct null peaks at or above each z-score per micrograph,
:math:`\lambda(t)`, and so the threshold for a target number of false positives:

1. lattice local maxima over position, in-plane angle, view direction and defocus
   plane (Worsley's discrete local maxima), from the field's correlations one and two
   grid steps along every axis;
2. the peak finder's suppression radius (crowding);
3. the per-pixel z-score (Panther's z is ``(max - mean) / sd`` over every hypothesis
   at the pixel), through ``kappa``, the mean squared correlation of random
   hypotheses.

This module supplies Panther's side of the calculation, from Panther's own pieces:

* the image: the prepared micrograph's search image (``W_avg * C_i^2``, zero mean,
  unit variance; :meth:`~panther_em.search.micrograph.Micrograph.prepare`);
* the templates (:class:`TemplateProjector`): the volume's projection exactly as the
  decomposition makes it (:func:`~panther_em.decomposition.projections.
  precompute_volume_dft`, :func:`~panther_em.decomposition.projections.
  project_from_precomputed_dft`), under the whitening's template filter
  ``W_avg * band``, normalized to unit variance before the CTF as the decomposition
  does, and masked to the polar grid's support disk;
* the CTF: the micrograph's own CTF planes (which carry the after-CTF plane factors
  ``b_d``), sampled on the lag grid of the template box (:func:`lag_grid_planes`).
  Panther applies the CTF to the image; for the covariance it is the same to apply
  it to the template;
* after-CTF normalization: every sampled template gets the row factor
  :math:`a = (\overline{e^{\mathrm{ref}}} / c)^{-1/2}` that
  :class:`~panther_em.search.normalization.AfterCtfNormalization` gives a template
  row: its noise variance under the run's reference noise (white noise under the
  ring-averaged ``CTF^2`` of every searched plane,
  :func:`~panther_em.search.normalization.reference_noise`), averaged over the
  planes (the constant ``c`` cancels in the z-score). The plane factors ``b_d`` are
  the micrograph's own, carried by its CTF planes. Computing ``a`` for each sampled
  template, rather than interpolating the grid rows' factors, equalizes the model's
  templates as Panther equalizes its rows on any grid;
* the grid: HEALPix ``nside`` x ``n_psi`` (:func:`lattice_neighbours`): in-plane
  steps of ``2 pi / n_psi`` about the body z axis and view steps of one HEALPix
  pixel, ``sqrt(pi / 3) / nside``, about the body x and y axes (the pixel lattice
  treated as locally square), each composed on the right of the rotation. A HEALPix
  x psi grid is uniform on SO(3), so its points are sampled as Haar-random
  rotations.

SVD rank truncation and the polar warp are not modelled (the field is that of the
exact templates); nor is the stage-1 gate, so the prediction is for a complete
search.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import roma
import torch
from leopard_em.analysis.analytic_null import (
    AnalyticNullPrediction,
    AxisNeighbours,
    PowerWeighting,
    estimate_kappa,
    estimate_mean_template,
    predict_null_rate,
    sample_lattice,
)

from panther_em.decomposition.projections import (
    apply_fourier_filters,
    precompute_volume_dft,
    project_from_precomputed_dft,
)
from panther_em.filters.curves import curve_to_grid
from panther_em.orientations import healpix_ipix

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from panther_em.search.micrograph import Micrograph

__all__ = [
    "DEFAULT_THRESHOLDS",
    "TemplateProjector",
    "haar_rotations",
    "healpix_nside",
    "lag_grid_planes",
    "lattice_neighbours",
    "predict_null",
    "reference_grid",
]

DEFAULT_THRESHOLDS = np.round(np.arange(2.0, 10.0, 0.01), 3)


def lag_grid_planes(
    planes: torch.Tensor,
    image_shape: tuple[int, int],
    lag_shape: tuple[int, int],
) -> torch.Tensor:
    """Real rfft-layout planes of an image grid, sampled on a coarser grid.

    Plane values at the frequencies of the ``lag_shape`` rfft grid, by bilinear
    interpolation between the image grid's frequencies (rows periodic). Exact when
    ``lag_shape`` divides ``image_shape``.

    Parameters
    ----------
    planes : torch.Tensor
        ``(D, H, W // 2 + 1)`` real planes (e.g. a micrograph's CTF planes).
    image_shape : tuple[int, int]
        ``(H, W)``.
    lag_shape : tuple[int, int]
        ``(Ly, Lx)`` of the grid to sample on.

    Returns
    -------
    torch.Tensor
        ``(D, Ly, Lx // 2 + 1)`` float32, on ``planes``' device.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    ly, lx = int(lag_shape[0]), int(lag_shape[1])
    if planes.dim() != 3 or tuple(planes.shape[1:]) != (height, width // 2 + 1):
        raise ValueError(
            f"planes must be (D, {height}, {width // 2 + 1}), got {tuple(planes.shape)}"
        )
    device = planes.device
    # lag row i is frequency (i mod ly) / ly, image row i * H / ly (mod H); columns
    # j / lx, image column j * W / lx
    fy = torch.arange(ly, dtype=torch.float64) * height / ly
    fx = torch.arange(lx // 2 + 1, dtype=torch.float64) * width / lx
    y0 = torch.floor(fy).long()
    x0 = torch.floor(fx).long().clamp(max=width // 2)
    ty = (fy - y0).to(device, torch.float32)[:, None]
    tx = (fx - x0).to(device, torch.float32)[None, :]
    y1 = ((y0 + 1) % height).to(device)
    x1 = (x0 + 1).clamp(max=width // 2).to(device)
    y0, x0 = y0.to(device), x0.to(device)
    p = planes.to(torch.float32)
    rows0, rows1 = p[:, y0], p[:, y1]
    out = (1 - ty) * ((1 - tx) * rows0[..., x0] + tx * rows0[..., x1]) + ty * (
        (1 - tx) * rows1[..., x0] + tx * rows1[..., x1]
    )
    return out


def healpix_nside(
    phi_deg: np.ndarray | torch.Tensor, theta_deg: np.ndarray | torch.Tensor
) -> int:
    """``nside`` of a decomposition's orientations, which must be a HEALPix grid.

    Raises unless the ``(O,)`` orientations (degrees) are the complete set of HEALPix
    pixel centers at some ``nside``, the grid the analytic null models.
    """
    grid = healpix_ipix(phi_deg, theta_deg)
    if grid is None or 12 * grid[0] ** 2 != int(np.asarray(phi_deg).size):
        raise ValueError(
            "the analytic null models a complete HEALPix grid of orientations; the "
            "decomposition's orientations are not one"
        )
    return int(grid[0])


def reference_grid(
    frequencies: torch.Tensor,
    reference: torch.Tensor,
    box: tuple[int, int],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """The after-CTF reference noise on a template box's rfft grid.

    Parameters
    ----------
    frequencies : torch.Tensor
        ``(F,)`` the curves' frequency axis (the error table's).
    reference : torch.Tensor
        ``(D, F)`` reference noise of every plane of the run
        (:func:`~panther_em.search.normalization.reference_noise`).
    box : tuple[int, int]
        Template box ``(h, w)``.
    device : torch.device or str, optional
        Where the grid is returned.

    Returns
    -------
    torch.Tensor
        ``(D, h, w // 2 + 1)`` float32.
    """
    return torch.stack(
        [
            curve_to_grid(curve.to(torch.float32), frequencies, box, device=device)
            for curve in reference
        ]
    )


def haar_rotations(n: int, generator: torch.Generator) -> torch.Tensor:
    """``(n, 3, 3)`` float32 Haar-random rotations (normalized Gaussian quaternions)."""
    quat = torch.randn((n, 4), generator=generator, dtype=torch.float64)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    rotation: torch.Tensor = roma.unitquat_to_rotmat(quat)
    return rotation.to(torch.float32)


def _axis_rotation(axis: int, angle: float) -> torch.Tensor:
    vec = torch.zeros(3, dtype=torch.float64)
    vec[axis] = angle
    rotation: torch.Tensor = roma.rotvec_to_rotmat(vec)
    return rotation


def lattice_neighbours(
    nside: int, n_psi: int
) -> Callable[[torch.Tensor], list[AxisNeighbours]]:
    """Neighbours on Panther's HEALPix ``nside`` x ``n_psi`` grid.

    For rotations ``R`` (``(n, 3, 3)``), the rotations one and two steps away along
    each axis, composed on the right (``R @ step``): in-plane angle ``psi`` (body z,
    ``2 pi / n_psi``) and the view direction along body x and y (``view_x``,
    ``view_y``: one HEALPix pixel, ``sqrt(pi / 3) / nside``). Every axis is periodic.
    """
    steps = (
        ("psi", 2, 2.0 * math.pi / n_psi),
        ("view_x", 0, math.sqrt(math.pi / 3.0) / nside),
        ("view_y", 1, math.sqrt(math.pi / 3.0) / nside),
    )

    def neighbours(rotations: torch.Tensor) -> list[AxisNeighbours]:
        out = []
        for name, axis, step in steps:
            shifted = [
                rotations @ _axis_rotation(axis, m * step).to(rotations)
                for m in (1, -1, 2, -2)
            ]
            out.append(
                AxisNeighbours(
                    name=name, steps=(shifted[0], shifted[1], shifted[2], shifted[3])
                )
            )
        return out

    return neighbours


class TemplateProjector:
    """Panther's search templates at arbitrary rotations, on a lag grid.

    For rotations ``R``, the template of every searched plane ``d``:
    ``a(R) * b_d * CTF_d * m_R`` on the lag grid of ``weighting``, DC removed, where
    ``m_R`` is the volume's projection as the decomposition makes it (padded volume
    DFT, Fourier slice, the template filter, unit variance when the decomposition
    normalized its projections) masked to ``support``; ``CTF_d`` (which carries
    ``b_d``) is given on the lag grid; ``a(R)`` is the after-CTF row factor of
    ``m_R`` under ``reference`` (:meth:`row_scale`; 1 without a reference).

    Parameters
    ----------
    volume : torch.Tensor or numpy.ndarray
        ``(d, d, d)`` template volume.
    template_filter : torch.Tensor
        ``(d, d // 2 + 1)`` the decomposition's Fourier filter (``W_avg * band``).
    weighting : PowerWeighting
        The image's lag-grid power weighting (sets the lag grid).
    ctf : torch.Tensor
        ``(D, Ly, Lx // 2 + 1)`` the searched CTF planes on the lag grid.
    support : torch.Tensor, optional
        ``(d, d)`` bool support disk of the polar grid.
    normalize : bool
        Unit-variance projections (the decomposition's ``normalized_projections``).
    reference : torch.Tensor, optional
        ``(D_ref, d, d // 2 + 1)`` the after-CTF reference noise of every plane of
        the run on the box's rfft grid (:func:`reference_grid`); ``None`` for
        ``before_ctf``.
    device : torch.device or str
        Where the projections are made.
    """

    def __init__(
        self,
        volume: torch.Tensor | np.ndarray,
        template_filter: torch.Tensor,
        weighting: PowerWeighting,
        ctf: torch.Tensor,
        *,
        support: torch.Tensor | None = None,
        normalize: bool = True,
        reference: torch.Tensor | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        vol = torch.as_tensor(np.asarray(volume, dtype=np.float32)).to(self.device)
        if vol.dim() != 3 or len(set(vol.shape)) != 1:
            raise ValueError(f"volume must be cubic, got {tuple(vol.shape)}")
        box = int(vol.shape[-1])
        self.box = (box, box)
        if tuple(template_filter.shape) != (box, box // 2 + 1):
            raise ValueError(
                f"template_filter must be ({box}, {box // 2 + 1}), got "
                f"{tuple(template_filter.shape)}"
            )
        self.dft, self.edge, self.pad = precompute_volume_dft(vol)
        del vol
        self.filter = template_filter.to(self.device, torch.float32)
        self.weighting = weighting
        lag = tuple(int(s) for s in weighting.lag_shape)
        if tuple(ctf.shape[1:]) != (lag[0], lag[1] // 2 + 1):
            raise ValueError(
                f"ctf must be (D, {lag[0]}, {lag[1] // 2 + 1}) on the lag grid, got "
                f"{tuple(ctf.shape)}"
            )
        self.ctf = ctf.to(self.device, torch.float32)
        self.support = (
            None if support is None else support.to(self.device, torch.float32)
        )
        self.normalize = bool(normalize)
        self.reference: torch.Tensor | None = None
        self.white = 1.0
        if reference is not None:
            if tuple(reference.shape[1:]) != (box, box // 2 + 1):
                raise ValueError(
                    f"reference must be (D, {box}, {box // 2 + 1}), got "
                    f"{tuple(reference.shape)}"
                )
            hermitian = torch.full((box // 2 + 1,), 2.0, device=self.device)
            hermitian[0] = 1.0
            if box % 2 == 0:
                hermitian[-1] = 1.0
            weights = reference.to(self.device, torch.float32).mean(dim=0) * hermitian
            self.reference = weights
            # a white, unit-variance template's |m^|^2 is h w: its factor is 1
            self.white = float(weights.sum()) * box * box

    @property
    def n_planes(self) -> int:
        """Searched CTF planes ``D``."""
        return int(self.ctf.shape[0])

    def row_scale(self, templates: torch.Tensor) -> torch.Tensor | None:
        """``(n,)`` after-CTF row factor of ``(n, d, d)`` templates (or ``None``).

        ``a = (e / c)^(-1/2)``, ``e`` the template's noise variance under the
        reference noise averaged over the run's planes, ``c`` that of a white
        unit-variance template.
        """
        if self.reference is None:
            return None
        power = torch.fft.rfft2(templates).abs().square()
        e = (power * self.reference).sum(dim=(-2, -1))
        scale: torch.Tensor = (e / self.white).rsqrt()
        return scale

    @torch.no_grad()
    def __call__(self, rotations: torch.Tensor) -> torch.Tensor:
        """``(D, n, Ly, Lx // 2 + 1)`` templates of ``(n, 3, 3)`` rotations."""
        rot = rotations.to(self.device, torch.float32)
        projections = project_from_precomputed_dft(self.dft, rot, self.edge, self.pad)
        m = apply_fourier_filters(projections, self.filter[None])[0]
        if self.normalize:
            var, mean = torch.var_mean(m, dim=(-2, -1), keepdim=True)
            m = (m - mean) / torch.sqrt(var + 1e-8)
        if self.support is not None:
            m = m * self.support
        scale = self.row_scale(m)
        if scale is not None:
            m = m * scale[:, None, None]
        q: torch.Tensor = self.weighting.transform(m)
        return q[None] * self.ctf[:, None]


def _batches(rotations: torch.Tensor, size: int) -> Iterator[torch.Tensor]:
    for i in range(0, int(rotations.shape[0]), size):
        yield rotations[i : i + size]


@dataclass(frozen=True)
class _Sampler:
    generator: torch.Generator

    def rotations(self, n: int) -> torch.Tensor:
        return haar_rotations(n, self.generator)

    def pairs(self, n_pairs: int, size: int) -> Iterator[tuple[Any, Any]]:
        for _ in range(n_pairs):
            yield self.rotations(size), self.rotations(size)


# pylint: disable=too-many-locals
def predict_null(
    micrograph: Micrograph,
    volume: torch.Tensor | np.ndarray,
    template_filter: torch.Tensor,
    *,
    nside: int,
    n_psi: int,
    valid_area: float,
    planes: Sequence[int] | None = None,
    n_filters: int = 1,
    support: torch.Tensor | None = None,
    normalize_projections: bool = True,
    reference: torch.Tensor | None = None,
    mask_radius: float = 10.0,
    n_mean_samples: int = 3000,
    n_kappa_pairs: int = 64,
    n_lattice_samples: int = 128,
    batch_size: int = 4,
    thresholds: np.ndarray | None = None,
    seed: int = 0,
    device: torch.device | str = "cpu",
    details: dict[str, Any] | None = None,
) -> AnalyticNullPrediction:
    """Predict the null peak rate of a Panther search of ``micrograph``.

    Parameters
    ----------
    micrograph : Micrograph
        The prepared micrograph (for after-CTF normalization, with its plane factors
        applied, as the search sees it). Only its search image's power spectrum and
        its CTF planes are used, so a phase-randomised copy gives the same answer.
    volume : torch.Tensor or numpy.ndarray
        ``(d, d, d)`` the template volume the decomposition was made from.
    template_filter : torch.Tensor
        ``(d, d // 2 + 1)`` the decomposition's Fourier filter
        (:meth:`~panther_em.filters.WhiteningFilter.template_filter`).
    nside, n_psi : int
        The searched HEALPix grid and in-plane angles (the final stage's).
    valid_area : float
        Search positions peaks are counted in (border and mask removed).
    planes : sequence of int, optional
        The searched CTF planes; default all of the micrograph's.
    n_filters : int
        Fourier filters of the decomposition (hypotheses per orientation); only 1
        is modelled.
    support : torch.Tensor, optional
        ``(d, d)`` bool support disk of the polar grid.
    normalize_projections : bool
        Whether the decomposition normalized its projections. Default True.
    reference : torch.Tensor, optional
        ``(D_ref, d, d // 2 + 1)`` after-CTF reference noise of the run's planes on
        the template box (:func:`reference_grid`); ``None`` for ``before_ctf``.
    mask_radius : float
        The peak finder's suppression radius, pixels. Default 10.
    n_mean_samples, n_kappa_pairs, n_lattice_samples : int
        Rotations for the mean template, pairs of rotation batches for ``kappa``,
        rotations for the lattice correlations (each with every plane).
    batch_size : int
        Rotations per batch for ``kappa`` and the lattice. Default 4.
    thresholds : numpy.ndarray, optional
        Thresholds to evaluate; default 2 to 10 in steps of 0.01.
    seed : int
        Seed of the sampled rotations.
    device : torch.device or str
        Where the projections and inner products are computed.
    details : dict, optional
        Extras recorded in the prediction's ``details``.

    Returns
    -------
    AnalyticNullPrediction
        ``.zscore_cutoff(fp)`` is the predicted threshold for ``fp`` null peaks per
        micrograph; ``.as_dict()`` the JSON record.
    """
    start = time.time()
    if n_filters != 1:
        raise NotImplementedError(
            "the analytic null models decompositions with one Fourier filter"
        )
    device = torch.device(device)
    image = micrograph.image
    image_shape = micrograph.image_shape
    box = (int(template_filter.shape[0]), int(template_filter.shape[0]))
    weighting = PowerWeighting(
        torch.fft.rfft2(image.to(device, torch.float32)), image_shape, box
    ).to(device)
    ctf = micrograph.ctf if planes is None else micrograph.ctf[list(planes)]
    lag_ctf = lag_grid_planes(ctf.to(device), image_shape, weighting.lag_shape)
    project = TemplateProjector(
        volume,
        template_filter,
        weighting,
        lag_ctf,
        support=support,
        normalize=normalize_projections,
        reference=reference,
        device=device,
    )
    n_planes = project.n_planes
    sampler = _Sampler(torch.Generator().manual_seed(int(seed)))

    per_batch = max(1, 16 // n_planes)
    mean = estimate_mean_template(
        project, weighting, _batches(sampler.rotations(n_mean_samples), per_batch)
    )
    kappa = estimate_kappa(
        project, weighting, mean.mean, sampler.pairs(n_kappa_pairs, batch_size)
    )
    lattice_batch = batch_size if n_planes > 1 else 2 * batch_size
    samples = sample_lattice(
        project,
        weighting,
        mean.mean,
        _batches(sampler.rotations(n_lattice_samples), lattice_batch),
        lattice_neighbours(nside, n_psi),
    )
    n_orientations = 12 * nside**2
    n_points = float(valid_area) * n_orientations * n_psi * n_filters * n_planes
    record: dict[str, Any] = {
        "engine": "panther_em.search",
        "nside": int(nside),
        "n_psi": int(n_psi),
        "orientations": n_orientations,
        "planes": n_planes,
        "hypotheses": n_orientations * n_psi * n_filters * n_planes,
        "normalization": "before_ctf" if reference is None else "after_ctf",
        "image_shape": list(image_shape),
        "box_shape": list(box),
        "lag_shape": list(weighting.lag_shape),
        "support_masked": support is not None,
        "n_mean_samples": int(n_mean_samples),
        "n_kappa_pairs": int(n_kappa_pairs),
        "n_lattice_samples": int(n_lattice_samples),
        "seed": int(seed),
        "weighting_negative_fraction": weighting.negative_fraction,
        "device": str(device),
    }
    record.update(details or {})
    record["seconds"] = time.time() - start
    prediction: AnalyticNullPrediction = predict_null_rate(
        samples,
        kappa,
        mean.rho_bar,
        n_points,
        valid_area,
        mask_radius,
        DEFAULT_THRESHOLDS if thresholds is None else thresholds,
        details=record,
    )
    return prediction
