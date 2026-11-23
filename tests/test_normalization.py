"""After-CTF normalization (``SearchConfig.normalization: after_ctf``).

Every ``(plane, template)`` hypothesis is scaled by ``s_dn = (e_dn / mean e)^-1/2``:
the factors equalize the predicted noise variance exactly; the dense loop (torch and
fused reductions, eager and graphed), the local pass (spectra and per-plane moment
forms) and the threshold test carry them; a search runs with them and still finds the
synthetic particle.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from conftest import read_group, requires_cuda
from noise_case import noise_case
from test_correlator import _fill_store, _reference_corr
from test_local import CASES, _pass

from panther_em.managers import SearchConfig, search
from panther_em.search import Micrograph, SearchResult, ThresholdTest
from panther_em.search.config import Tuning
from panther_em.search.correlation import Correlator, LocalWork, LoopPlan
from panther_em.search.correlation.contraction import BatchShape
from panther_em.search.correlation.moments import (
    MomentForms,
    block_moments,
    moments_to_stats,
)
from panther_em.search.hypotheses import HypothesisSpace
from panther_em.search.normalization import (
    hypothesis_scale,
    hypothesis_variance,
    normalize,
    relative_sd,
    template_power,
)
from panther_em.search.planning import DeviceInfo, plan_memory
from panther_em.search.stage import Stage
from panther_em.search.weights import WeightSource

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:ComplexHalf support is experimental"),
]

RECTS = [(0, 0, 3, 4)]
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "threshold_test_pipeline.py"


@pytest.fixture(scope="module")
def case():
    case = noise_case()
    table = case.table(RECTS)
    return case, table, case.images(table)


def _random_scale(n_defocus: int, n_templates: int, seed: int) -> torch.Tensor:
    """A non-separable ``(D, N)`` float64 factor in ``[0.6, 1.6]``."""
    gen = torch.Generator().manual_seed(seed)
    return 0.6 + torch.rand(n_defocus, n_templates, generator=gen, dtype=torch.float64)


# --------------------------------------------------------------------------- #
# The factors
# --------------------------------------------------------------------------- #


def test_hypothesis_scale_equalizes_exactly():
    gen = torch.Generator().manual_seed(0)
    e = 1.0 + 0.3 * torch.rand(5, 40, generator=gen, dtype=torch.float64)
    s = hypothesis_scale(e)
    scaled = e * s.square()
    assert relative_sd(e) > 0.05
    assert relative_sd(scaled) < 1e-12
    assert float(scaled.mean()) == pytest.approx(float(e.mean()), rel=1e-12)
    with pytest.raises(ValueError, match="positive"):
        hypothesis_scale(torch.zeros(2, 3, dtype=torch.float64))


def test_q_kept_is_the_hypothesis_variance(case):
    _, table, images = case
    q = ThresholdTest(table).scales(images).q_kept
    e = hypothesis_variance(images.noise, template_power(table))
    torch.testing.assert_close(e, q, rtol=1e-5, atol=0)


def test_normalize_sets_the_scale(case):
    _, table, images = case
    scaled, report = normalize(images, table)
    e = hypothesis_variance(images.noise, template_power(table))
    torch.testing.assert_close(scaled.hypothesis_scale, (e / e.mean()).rsqrt())
    assert report["variance_rel_sd_before"] == pytest.approx(relative_sd(e))
    # the image, CTF planes and curves are untouched
    for name in ("image", "ctf", "noise", "signal", "level"):
        assert torch.equal(getattr(scaled, name), getattr(images, name))
    assert images.hypothesis_scale is None


def test_threshold_test_scales_with_the_hypotheses(case):
    _, table, images = case
    s = _random_scale(images.n_defocus, table.n_templates, 1)
    test = ThresholdTest(table)
    before = test.scales(images)
    after = test.scales(Micrograph(**{**vars(images), "hypothesis_scale": s}))
    torch.testing.assert_close(after.q_kept, before.q_kept * s.square())
    torch.testing.assert_close(after.q_perp, before.q_perp * s.square())
    torch.testing.assert_close(after.lam, before.lam)
    torch.testing.assert_close(after.rho, before.rho)
    assert after.qbar_perp == pytest.approx(float(after.q_perp.mean()))
    with pytest.raises(ValueError, match="hypothesis_scale"):
        test.scales(
            Micrograph(**{**vars(images), "hypothesis_scale": s[:, :-1].contiguous()})
        )


def test_micrograph_validates_and_saves_the_scale(case, tmp_path):
    _, table, images = case
    s = _random_scale(images.n_defocus, table.n_templates, 2)
    scaled = Micrograph(**{**vars(images), "hypothesis_scale": s})
    scaled.save(tmp_path / "m.h5")
    loaded = Micrograph.load(tmp_path / "m.h5")
    assert torch.equal(loaded.hypothesis_scale, s)
    with pytest.raises(ValueError, match="finite and positive"):
        Micrograph(**{**vars(images), "hypothesis_scale": -s})
    with pytest.raises(ValueError, match="float64 CPU"):
        Micrograph(**{**vars(images), "hypothesis_scale": s.float()})


# --------------------------------------------------------------------------- #
# The dense loop
# --------------------------------------------------------------------------- #


def _scaled_reference(result, spec, features, hyps, n_psi, scale):
    """Dense statistics of the scaled correlograms ``s_dn c_dn(psi)``."""
    corr = _reference_corr(result, spec, features, hyps, n_psi)  # (P, D, N, psi)
    corr = corr * hyps.select(scale).to(corr.dtype)[None, :, :, None]
    flat = corr.reshape(corr.shape[0], -1)
    mip, amax = flat.max(dim=1)
    mean = flat.mean(dim=1)
    var = (flat**2).mean(dim=1) - mean**2
    return mip, mean, var, amax // n_psi, (n_psi - amax % n_psi) % n_psi


def _dense(ctx, spec, store, space, *, fused, graph, template_ids=None, scale=None):
    memory = plan_memory(
        DeviceInfo.probe(ctx.device),
        spec,
        n_pixels=store.n_pixels,
        space=space,
        template_ids=template_ids,
        tuning=Tuning(
            batch=BatchShape(4, 4),
            block_pixels=store.block_pixels,
            use_cuda_graph=graph,
            use_fused_kernel=fused,
        ),
    )
    plan = LoopPlan.create(spec, store, memory)
    return plan, ctx.run(store, plan, scale=scale)


def _assert_scaled(maps, ref, tol):
    mip, mean, var, best_virtual, best_psi = ref
    torch.testing.assert_close(maps.mip.cpu(), mip, **tol)
    torch.testing.assert_close(maps.mean.cpu(), mean, **tol)
    torch.testing.assert_close(maps.variance.cpu(), var, **tol)
    assert torch.equal(maps.best_virtual.cpu(), best_virtual)
    assert torch.equal(maps.best_psi.cpu(), best_psi)


@pytest.mark.parametrize("template_ids", [None, torch.tensor([0, 2, 3, 5, 7])])
def test_scaled_dense_loop_matches_the_scaled_reference(synthetic_result, template_ids):
    result = synthetic_result(n_orientations=8)
    space = HypothesisSpace.from_result(result, n_defocus=3)
    spec = Stage.create([(0, 0, 4, 3), (0, 3, 2, 2)], n_psi=16)
    store, features = _fill_store(result, spec, 3, 8, "cpu")
    ctx = Correlator("cpu", WeightSource(result, "cpu"))
    scale = _random_scale(3, space.n_templates, 3)
    plan, maps = _dense(
        ctx,
        spec,
        store,
        space,
        fused=False,
        graph=False,
        template_ids=template_ids,
        scale=scale,
    )
    ref = _scaled_reference(result, spec, features, plan.hyps, 16, scale)
    _assert_scaled(maps, ref, {"rtol": 1e-4, "atol": 1e-4})
    # the cached loop is reset to ones for an unscaled pass
    _, plain = _dense(
        ctx, spec, store, space, fused=False, graph=False, template_ids=template_ids
    )
    ones = torch.ones_like(scale)
    _assert_scaled(
        plain,
        _scaled_reference(result, spec, features, plan.hyps, 16, ones),
        {"rtol": 1e-4, "atol": 1e-4},
    )


@requires_cuda
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
@pytest.mark.parametrize("graph", [False, True])
def test_scaled_fused_loop_matches_the_scaled_reference(
    synthetic_result, precision, graph
):
    device = torch.device("cuda")
    result = synthetic_result(n_orientations=8)
    space = HypothesisSpace.from_result(result, n_defocus=2)
    spec = Stage.create([(0, 0, 4, 3), (0, 3, 2, 2)], precision=precision, n_psi=128)
    store, features = _fill_store(result, spec, 2, 8, device)
    ctx = Correlator(device, WeightSource(result, device))
    tol = (
        {"rtol": 1e-4, "atol": 1e-4}
        if precision == "fp32"
        else {"rtol": 2e-2, "atol": 5e-2}
    )
    # two image sets through one (captured) loop: the buffer follows the factors
    for seed in (4, 5):
        scale = _random_scale(2, space.n_templates, seed)
        plan, maps = _dense(
            ctx, spec, store, space, fused=True, graph=graph, scale=scale
        )
        assert plan.fused
        assert (ctx.loops[plan.signature].graph is not None) == graph
        ref = _scaled_reference(result, spec, features, plan.hyps, 128, scale)
        mip, mean, var, best_virtual, _ = ref
        torch.testing.assert_close(maps.mip.cpu(), mip, **tol)
        torch.testing.assert_close(maps.mean.cpu(), mean, **tol)
        torch.testing.assert_close(maps.variance.cpu(), var, **tol)
        if precision == "fp32":
            assert torch.equal(maps.best_virtual.cpu(), best_virtual)
    assert len(ctx.loops) == 1


@requires_cuda
def test_unscaled_fused_loop_is_unchanged_by_the_scale_buffer(synthetic_result):
    """Ones in the buffer reproduce the unscaled kernel bit for bit."""
    device = torch.device("cuda")
    result = synthetic_result(n_orientations=8)
    space = HypothesisSpace.from_result(result, n_defocus=2)
    spec = Stage.create([(0, 0, 4, 3)], precision="fp32", n_psi=128)
    store, _ = _fill_store(result, spec, 2, 8, device)
    ctx = Correlator(device, WeightSource(result, device))
    _, plain = _dense(ctx, spec, store, space, fused=True, graph=False)
    ones = torch.ones(2, space.n_templates, dtype=torch.float64)
    _, unit = _dense(ctx, spec, store, space, fused=True, graph=False, scale=ones)
    for name in ("mip", "mean", "variance", "best_virtual", "best_psi"):
        assert torch.equal(getattr(plain, name), getattr(unit, name))


# --------------------------------------------------------------------------- #
# Local passes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("rects", "n_psi", "orientations"), CASES)
def test_scaled_moment_forms_equal_the_scaled_loop(
    synthetic_result, rects, n_psi, orientations
):
    template, plan, store = _pass(
        synthetic_result, rects, n_psi, orientations=orientations
    )
    space = plan.hyps.space
    scale = _random_scale(space.n_defocus, space.n_templates, 6)
    maps = Correlator("cpu", template.weights).run(store, plan, scale=scale)
    forms = MomentForms.build(
        template.weights,
        plan.layout,
        plan.engine,
        n_psi,
        template_rows=None if plan.hyps.is_contiguous else plan.hyps.template_rows(),
        plane_scale=plan.hyps.select(scale),
    )
    assert forms.n_form_planes == plan.hyps.n_defocus_selected
    for block in store.blocks(plan.groups):
        start, n = block.col_range[0], block.n_valid
        s1, s2 = block_moments(plan, forms, block.operands, n)
        mean, var = moments_to_stats(s1, s2, plan.hyps.n_evaluated)
        ref_var = maps.variance[start : start + n]
        torch.testing.assert_close(var, ref_var, rtol=2e-4, atol=0)
        assert (
            (mean - maps.mean[start : start + n]).abs() <= 2e-4 * ref_var.sqrt()
        ).all()


@pytest.mark.parametrize(("rects", "n_psi", "orientations"), CASES)
def test_scaled_local_correlation_matches_brute_force(
    synthetic_result, rects, n_psi, orientations
):
    template, plan, store = _pass(
        synthetic_result, rects, n_psi, orientations=orientations
    )
    space = plan.hyps.space
    scale = _random_scale(space.n_defocus, space.n_templates, 7)
    gen = torch.Generator().manual_seed(3)
    n_sel = plan.hyps.n_selected
    work = LocalWork(
        cells=torch.randint(0, 4, (store.n_pixels,), generator=gen),
        rows=torch.randint(0, n_sel, (4, 3), generator=gen),
    )
    correlator = Correlator("cpu", template.weights)
    maps = correlator.run_local(store, plan, work, chunk_bytes=1 << 30, scale=scale)
    ref = _scaled_local_reference(template, plan, store, work, plan.hyps.select(scale))
    torch.testing.assert_close(
        maps.mip, torch.tensor([r[0] for r in ref]), rtol=1e-5, atol=1e-5
    )
    assert maps.best_virtual.tolist() == [r[1] for r in ref]
    assert maps.best_psi.tolist() == [r[2] for r in ref]
    # the whole set's (scaled) moments, from per-plane forms
    full = correlator.run(store, plan, scale=scale)
    torch.testing.assert_close(maps.variance, full.variance, rtol=2e-4, atol=0)
    assert maps.n_evaluated == full.n_evaluated


def _scaled_local_reference(template, plan, store, work, selected):
    """Per column: the maximum of ``s_dn c_dn(psi)`` over its group (explicit irfft)."""
    hyps = plan.hyps
    n_psi, f = plan.stage.n_psi, plan.n_freq
    us = template.weights.us.to(torch.complex64)
    rows = hyps.template_rows(us.device)
    out = []
    for p in range(store.n_pixels):
        j = work.rows[int(work.cells[p])]
        w = us[rows[j]]  # (W, K, M)
        c = torch.zeros(hyps.n_defocus_selected, j.numel(), f, dtype=torch.complex64)
        for rect in plan.layout.rectangles:
            k0, m0, ke, me = rect
            group = store.group(rect)
            for d in range(hyps.n_defocus_selected):
                y = read_group(store, group, d)[p]  # (ke, me)
                c[d, :, k0 : k0 + ke] += torch.einsum(
                    "km,wkm->wk", y, w[:, k0 : k0 + ke, m0 : m0 + me].conj()
                )
        c[..., 0] = c[..., 0].real
        if n_psi % 2 == 0 and f == n_psi // 2 + 1:
            c[..., -1] = c[..., -1].real
        corr = torch.fft.irfft(c, n=n_psi, dim=-1, norm="forward")
        corr = corr * selected[:, j].to(corr.dtype)[:, :, None]
        value, flat = corr.reshape(-1).max(dim=0)
        d, rest = divmod(int(flat), j.numel() * n_psi)
        slot, raw = divmod(rest, n_psi)
        out.append(
            (float(value), d * hyps.n_selected + int(j[slot]), (n_psi - raw) % n_psi)
        )
    return out


# --------------------------------------------------------------------------- #
# The program
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("threshold_test_pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path_factory.mktemp("pipeline")
    module.run(out)
    return module, out


def test_default_is_after_ctf(pipeline):
    _, out = pipeline
    config = SearchConfig.from_yaml(out / "search.yaml")
    assert config.normalization == "after_ctf"
    data = config.model_dump()
    del data["normalization"]  # a config written before the option existed
    assert SearchConfig.model_validate(data).normalization == "after_ctf"


def test_before_ctf_sets_no_scale(pipeline, tmp_path):
    _, out = pipeline
    config = SearchConfig.from_yaml(out / "search.yaml")
    config = config.model_copy(
        update={"output_dir": tmp_path, "normalization": "before_ctf"}
    )
    state = search._Run(config)
    assert state.normalization_table is None
    entry = config.micrographs[0]
    assert state.micrograph(entry).hypothesis_scale is None
    state.release()


def test_after_ctf_search(pipeline, tmp_path):
    module, out = pipeline
    base = SearchConfig.from_yaml(out / "search.yaml")
    config = base.model_copy(
        update={"output_dir": tmp_path / "after", "normalization": "after_ctf"}
    )
    state = search._Run(config)
    micrograph = state.micrograph(config.micrographs[0])
    table = state.normalization_table
    e = hypothesis_variance(micrograph.noise, template_power(table))
    torch.testing.assert_close(micrograph.hypothesis_scale, (e / e.mean()).rsqrt())
    state.release()

    done = search.run(config)
    assert sorted(done) == list(module.NAMES)
    summary = json.loads((tmp_path / "after" / "search_summary.json").read_text())
    old_way = search.run(
        base.model_copy(
            update={"output_dir": tmp_path / "before", "normalization": "before_ctf"}
        )
    )
    for name in module.NAMES:
        record = summary[name]["normalization"]
        assert record["variance_rel_sd_before"] > 0
        assert 0 < record["scale"]["min"] <= 1 <= record["scale"]["max"]

    # the synthetic particle is found as before: same label, orientation and plane,
    # z within a few percent (its hypothesis's factor is close to 1)
    p = module.PARTICLE
    r, c = p["top_left"]
    before = SearchResult.load(old_way[p["micrograph"]])
    after = SearchResult.load(done[p["micrograph"]])
    assert int(after.label[0, r, c]) == int(before.label[0, r, c])
    for key in ("orientation_index", "defocus_index"):
        assert int(after.decode()[key][0, r, c]) == int(before.decode()[key][0, r, c])
    z0, z1 = float(before.zscore[0, r, c]), float(after.zscore[0, r, c])
    assert z1 == pytest.approx(z0, rel=0.1)
