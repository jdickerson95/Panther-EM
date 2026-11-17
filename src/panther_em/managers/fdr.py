r"""False-discovery-rate detection threshold, set per micrograph after the search.

The search's default (``SearchConfig.fdr = 0.05``): instead of one fixed z threshold
that gives one expected false detection per micrograph, the final picks are the peaks
above the Benjamini-Hochberg threshold at false discovery rate ``q``.

Null model. The expected number of false peaks above ``t`` is the independent-trial
count of the search, :math:`\lambda(t) = N_\mathrm{pos} N_\mathrm{hyp} Q(t)` (the same
formula that sets the 1-FP threshold, :func:`~panther_em.search.zscore_threshold`). The
search's trials are correlated, so the formula over-counts false peaks: on xenon_131
with after-CTF normalization the measured null is 0.37 (B 0) to 0.14 (B 120) of it. The
threshold is therefore conservative -- the realised FDR is below ``q`` -- and needs no
null micrographs. (With phase-randomised nulls a calibrated cut is lower still; see
``docs/after_ctf_normalization.md``.)

Rule (step-up, :math:`\pi_0 = 1`). With the peak z-scores sorted in decreasing order,
:math:`z_{(1)} \ge z_{(2)} \ge \dots`, the threshold is :math:`z_{(k)}` for the largest
``k`` with :math:`\lambda(z_{(k)}) \le q k`; no such ``k``: no picks.

Peaks are local maxima at least ``peak_radius`` apart among the pixels the final stage
ACCEPTED, found as the particle export finds them (``find_peaks_from_zscore``), so the
count ``k`` is the number of particles the export would write.

Reach. The threshold is only known after the search, but the stages decide per pixel
during it, so with ``fdr`` set the final stage's ``"auto"`` threshold is lowered by
``fdr_reach`` (0.5) z below the 1-FP threshold -- the floor of the FDR cut -- and the
earlier stages' (the stage-1 gate) by ``fdr_gate_reach`` (0.25): the gate keeps pixels
about 0.3 z below its own threshold, so the final stage still evaluates exactly the
pixels an FDR cut down to the floor could pick, at 1.6 to 1.9 times the search time
instead of the ~6x a gate lowered by the full 0.5 would cost. A BH threshold below
the floor is clamped to it (``capped`` in the report): with the formula null that
takes about :math:`e^{T \cdot 0.5} / q \approx 1000` peaks per micrograph (T = 7.9).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from scipy.special import ndtr

from panther_em.search import PixelLabel

if TYPE_CHECKING:
    from panther_em.search import SearchResult


def expected_false_peaks(
    z: torch.Tensor, n_positions: int, n_hypotheses: int
) -> torch.Tensor:
    r""":math:`\lambda(z) = N_\mathrm{pos} N_\mathrm{hyp} Q(z)`, float64."""
    sf = torch.from_numpy(ndtr(-z.double().cpu().numpy()))  # Q(z), accurate far out
    return float(n_positions) * float(n_hypotheses) * sf


def bh_threshold(
    peak_z: torch.Tensor, n_positions: int, n_hypotheses: int, q: float
) -> float | None:
    """Benjamini-Hochberg threshold of ``peak_z`` at FDR ``q`` (``None``: no picks)."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1)")
    z = torch.sort(peak_z.double().cpu().reshape(-1), descending=True).values
    z = z[torch.isfinite(z)]
    if z.numel() == 0:
        return None
    k = torch.arange(1, z.numel() + 1, dtype=torch.float64)
    ok = torch.nonzero(expected_false_peaks(z, n_positions, n_hypotheses) <= q * k)
    if ok.numel() == 0:
        return None
    return float(z[int(ok[-1])])


def _disk_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """``(H, W)`` bool: within ``radius`` (Euclidean) of a True pixel of ``mask``."""
    r = int(radius)
    yy, xx = torch.meshgrid(
        torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing="ij"
    )
    kernel = ((yy * yy + xx * xx) <= r * r).to(torch.float32)[None, None]
    hit = torch.nn.functional.conv2d(
        mask.to(torch.float32)[None, None], kernel.to(mask.device), padding=r
    )
    return hit[0, 0] > 0.5


def apply_fdr(
    result: SearchResult,
    *,
    q: float,
    n_positions: int,
    n_hypotheses: int,
    floor: float,
    peak_radius: float,
    absorb_radius: int = 0,
) -> tuple[SearchResult, dict]:
    """``result`` with only the pixels at or above the BH threshold left ACCEPTED.

    Parameters
    ----------
    result : SearchResult
        A one-micrograph result whose final stage accepted every pixel above
        ``floor``.
    q : float
        False discovery rate.
    n_positions, n_hypotheses : int
        The search's positions per micrograph and hypotheses per position (the
        1-FP threshold's counts).
    floor : float
        The lowered stage threshold; a BH threshold below it is clamped to it.
    peak_radius : float
        Minimum distance between peaks, pixels (the export's ``peak_radius``).
    absorb_radius : int
        The final stage's absorb radius: ABSORBED pixels no longer within it of an
        ACCEPTED pixel go back to AMBIGUOUS.

    Returns
    -------
    tuple
        The relabelled result and a report: the BH threshold (``None`` if no
        peak qualifies), the applied threshold, whether it was capped at the floor,
        the peaks considered and picked, and the expected false picks under the
        formula.
    """
    from leopard_em.analysis.zscore_metric import find_peaks_from_zscore

    if int(result.label.shape[0]) != 1:
        raise ValueError("apply_fdr takes a one-micrograph result")
    label = result.label[0]
    zscore = result.zscore[0].detach().to(torch.float32)
    accepted = label == int(PixelLabel.ACCEPTED)
    masked = torch.where(accepted, zscore, torch.full_like(zscore, float("-inf"))).cpu()
    pos_y, pos_x = find_peaks_from_zscore(masked, float("-inf"), peak_radius)
    peak_z = masked[pos_y, pos_x] if pos_y.numel() else torch.zeros(0)
    peak_z = peak_z[torch.isfinite(peak_z)]
    t_bh = bh_threshold(peak_z, n_positions, n_hypotheses, q)
    threshold = float("inf") if t_bh is None else max(t_bh, floor)
    capped = t_bh is not None and t_bh < floor

    new = label.clone()
    drop = accepted & (zscore.to(label.device) < threshold)
    new[drop] = int(PixelLabel.REJECTED)
    if absorb_radius > 0:
        absorbed = new == int(PixelLabel.ABSORBED)
        if bool(absorbed.any()):
            near = _disk_dilate(new == int(PixelLabel.ACCEPTED), absorb_radius)
            new[absorbed & ~near] = int(PixelLabel.AMBIGUOUS)
    labels = result.label.clone()
    labels[0] = new
    n_picks = int((peak_z >= threshold).sum())
    report = {
        "q": q,
        "bh_threshold": t_bh,
        "threshold": None if t_bh is None else threshold,
        "floor": floor,
        "capped": capped,
        "n_peaks_considered": int(peak_z.numel()),
        "n_picks": n_picks,
        "expected_false_picks_formula": (
            None
            if t_bh is None
            else float(
                expected_false_peaks(
                    torch.tensor([threshold]), n_positions, n_hypotheses
                )[0]
            )
        ),
    }
    return dataclasses.replace(result, label=labels), report
