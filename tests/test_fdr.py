"""The FDR detection rule (``SearchConfig.fdr``, default 0.05)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from scipy.special import ndtri

from panther_em.managers import SearchConfig, search
from panther_em.managers.fdr import bh_threshold, expected_false_peaks
from panther_em.search import PixelLabel, SearchResult

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "threshold_test_pipeline.py"


def test_expected_false_peaks_is_the_formula():
    z = torch.tensor([7.0, 8.0])
    lam = expected_false_peaks(z, 1000, 10**9)
    t1 = float(-ndtri(1.0 / (1000 * 10**9)))  # the 1-FP threshold
    assert float(expected_false_peaks(torch.tensor([t1]), 1000, 10**9)[0]) == (
        pytest.approx(1.0, rel=1e-6)
    )
    assert float(lam[0]) > float(lam[1])


def test_bh_threshold_step_up():
    n_pos, n_hyp = 1000, 10**9
    t1 = float(-ndtri(1.0 / (n_pos * n_hyp)))
    # 40 strong peaks well above the 1-FP cut, then noise-like peaks below it
    strong = torch.full((40,), t1 + 1.0)
    weak = torch.linspace(t1 - 0.05, t1 - 2.0, 200)
    t = bh_threshold(torch.cat([strong, weak]), n_pos, n_hyp, 0.05)
    assert t is not None
    k = int((torch.cat([strong, weak]) >= t).sum())
    lam = float(expected_false_peaks(torch.tensor([t]), n_pos, n_hyp)[0])
    assert lam <= 0.05 * k + 1e-9  # the rule holds at the threshold
    # step-up: no lower peak also satisfies it
    lower = torch.cat([strong, weak])
    lower = lower[lower < t]
    for z in lower[:20]:
        kz = int((torch.cat([strong, weak]) >= z).sum())
        assert float(expected_false_peaks(z[None], n_pos, n_hyp)[0]) > 0.05 * kz
    assert t < t1  # many true peaks: the FDR cut sits below the 1-FP cut


def test_bh_threshold_no_picks_and_validation():
    assert bh_threshold(torch.tensor([3.0, 2.0]), 1000, 10**9, 0.05) is None
    assert bh_threshold(torch.zeros(0), 1000, 10**9, 0.05) is None
    with pytest.raises(ValueError):
        bh_threshold(torch.tensor([9.0]), 1000, 10**9, 0.0)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("threshold_test_pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path_factory.mktemp("pipeline")
    module.run(out)
    return module, out


def test_thresholds_lowered_only_when_active(pipeline, tmp_path):
    _, out = pipeline
    base = SearchConfig.from_yaml(out / "search.yaml")
    assert base.fdr == 0.05
    on = search._Run(base.model_copy(update={"output_dir": tmp_path}))
    off = search._Run(base.model_copy(update={"output_dir": tmp_path, "fdr": None}))
    n_pixels = on.n_positions((96, 96))
    for spec in base.stages:
        if spec.decision.threshold == "auto":
            reach = base.fdr_reach if spec is base.stages[-1] else base.fdr_gate_reach
            assert on.threshold(spec, n_pixels) == pytest.approx(
                off.threshold(spec, n_pixels) - reach
            )
    assert on.fdr_active and not off.fdr_active
    on.release()
    off.release()


def test_fdr_search(pipeline, tmp_path):
    module, out = pipeline
    base = SearchConfig.from_yaml(out / "search.yaml")
    done = search.run(base.model_copy(update={"output_dir": tmp_path / "fdr"}))
    summary = json.loads((tmp_path / "fdr" / "search_summary.json").read_text())
    for name in module.NAMES:
        rep = summary[name]["fdr"]
        assert rep["q"] == 0.05
        result = SearchResult.load(done[name])
        accepted = result.label[0] == int(PixelLabel.ACCEPTED)
        if rep["threshold"] is None:
            assert not bool(accepted.any())
            continue
        assert rep["threshold"] >= rep["floor"]
        assert bool((result.zscore[0][accepted] >= rep["threshold"] - 1e-6).all())
    # the synthetic particle is still picked
    p = module.PARTICLE
    r, c = p["top_left"]
    result = SearchResult.load(done[p["micrograph"]])
    assert int(result.label[0, r, c]) == int(PixelLabel.ACCEPTED)

    # fdr None: no report, thresholds as before
    plain = search.run(
        base.model_copy(update={"output_dir": tmp_path / "plain", "fdr": None})
    )
    summary = json.loads((tmp_path / "plain" / "search_summary.json").read_text())
    assert all("fdr" not in summary[name] for name in module.NAMES)
    assert sorted(plain) == list(module.NAMES)
