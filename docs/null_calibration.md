# Null calibration (`panther-em null`)

**Status:** new program. It needs the null modules of Leopard-EM's `jd_null_calibration`
branch (`leopard_em.utils.phase_randomization`, `leopard_em.analysis.empirical_null`,
`leopard_em.analysis.analytic_null`); until a Leopard-EM release contains them, the pin
`leopard-em==1.3.1` cannot import this program.

> **Why.** The `auto` threshold is the independent-trials formula: one expected false
> detection per micrograph if every `(position, plane, template, psi)` correlation were an
> independent Gaussian trial. They are not, and with after-CTF normalization (the default)
> the formula sits 0.13 z (B 0) to 0.28 z (B 120) above the real 1-false-positive threshold
> on xenon_131: safe, but it throws away particles, and it cannot give a false discovery
> rate. This program measures the real null of a search, or predicts it, so thresholds
> come from the null instead.

**In one sentence:** `panther-em null` searches phase-randomised copies of each micrograph
exactly as the micrograph is searched and sets the threshold for a target number of false
positives per micrograph from their pooled distinct peaks, and/or predicts the same null
peak rate analytically from the template, CTF, whitening and image spectrum, without a
search.

---

## 1. The problem: correlated tests and a per-pixel z

Panther reports, per position, `z = (max - mean) / sd` over the ~41 M hypotheses of the
13-plane production search (13 planes x 12,288 views x 256 psi). The formula threshold
`Q^-1(1 / (positions x hypotheses))` (`zscore_threshold`) treats those as independent
trials. Three effects move the real null away from it:

* **Correlated tests.** Neighbouring psi (r ~ 0.72), views (~ 0.78), defocus planes 200 A
  apart (~ 0.95) and pixels (~ 0.87) are strongly correlated, so there are far fewer
  effective trials: fewer null peaks than the formula counts.
* **Per-pixel normalization.** The sd at a pixel is estimated from the same correlations; a
  null peak raises its own pixel's sd, which thins the tail further.
* **Unequal hypothesis variances** fatten the tail. Before after-CTF normalization
  (`before_ctf`) this happened to cancel the other two at B 0 (formula and null within 0.01
  z); after it, the formula is conservative by 0.13 to 0.28 z.

None of this depends on particles, so it can be measured on micrographs that have none.

## 2. What is added

**Measured null** (`mode: empirical`, default). For every listed micrograph of a search
config and every seed:

1. the micrograph's MRC is phase-randomised: its Fourier amplitudes are kept exactly, every
   phase is replaced by the phase of seeded white noise
   (`leopard_em.utils.phase_randomization.phase_randomize_image`), then centred (and
   negated with `invert_contrast`) exactly as the search reads micrographs;
2. the copy is searched **under the micrograph's own name** with the unchanged search
   config, through the search program's own `_Run`: the micrograph's whitening row (the
   copy has the same amplitude spectrum, so `WhiteningFilter.check_image` passes), CTF
   planes, after-CTF plane factors `b_d`, stages, gate and thresholds;
3. the distinct peaks of the copy's final z-score map are kept
   (`leopard_em.analysis.empirical_null.distinct_peaks`: Leopard-EM's greedy peak finder,
   radius `peak_radius`, `border`, positions outside the micrograph's mask excluded), counted
   **only where the final stage scored the pixel** (`last_stage`): pixels a gate dropped keep
   an earlier stage's z, which is not the reported score.

The threshold for `f` false positives per micrograph is the `floor(R f)`-th highest of the
`R` copies' pooled peaks (`empirical_zscore_cutoff`, rule `mean`: unbiased for the target;
`confidence` gives at most the target with probability `confidence`). Its sd is about
`0.1 / sqrt(R)` z at 1 false positive: use 5 or more copies for a 1-false-positive target,
1 or 2 for targets of 10 or more.

**The gate.** Stages before the final one (the production stage 1) drop pixels their
threshold test predicts cannot reach `zeta * T`; below that the final map, and so the null
count, can be incomplete. The summary records this floor (`gate.floor`: 7.704 for the
production config, `0.98 x 7.861`) and flags every threshold `complete: true / false`
against it. A cut below the floor is biased low; check it with a copy searched without the
gate (stage 1 `threshold: -1e9, accept: false, calibration: null`).

**Analytic null** (`mode: analytic`, or `both`; `panther_em.search.null_model`). The
correlation of a phase-randomised image with the templates is a Gaussian random field whose
covariance is fixed by the templates and the image power spectrum. From that covariance
Leopard-EM's engine-agnostic analytic null (`leopard_em.analysis.analytic_null`) predicts
lambda(t), the expected distinct null peaks at or above t per micrograph: lattice local
maxima over position, psi, view and defocus plane (Worsley's discrete local maxima),
crowding by the peak finder's radius, and the per-pixel z through `kappa`, the mean squared
correlation of random hypotheses. Panther supplies its side, from its own pieces:

| piece | from |
|---|---|
| image power | the prepared micrograph's search image (`Micrograph.prepare`: `W_avg * C_i^2`, unit variance) |
| templates | the volume projected exactly as the decomposition does (`precompute_volume_dft`, `project_from_precomputed_dft`), under `WhiteningFilter.template_filter` (`W_avg * band`), unit variance before the CTF when the decomposition normalized its projections, masked to the polar grid's support disk |
| CTF | the micrograph's own CTF planes (with `b_d`), sampled on the template's lag grid (`lag_grid_planes`; exact when the lag grid divides the image, as 1024 divides 4096). Panther applies the CTF to the image; for the covariance that is the same as applying it to the template |
| `before_ctf` | nothing more: each hypothesis keeps the variance its CTF gives it |
| `after_ctf` | each sampled template gets the row factor `a = (e_ref / c)^(-1/2)` Panther gives a template row (its noise variance under the run's reference noise, `reference_noise`), and the micrograph's `b_d`. At 2,048 rows of the xenon_131 grid this `a` matches Panther's `a_n` to 0.25 % (correlation 0.994) |
| grid | HEALPix `nside` (from the decomposition's orientations) x `n_psi` (the final stage's): psi steps `2 pi / n_psi` about body z, view steps of one HEALPix pixel, `sqrt(pi / 3) / nside`, about body x and y, composed on the right; grid points sampled as Haar-random rotations (a HEALPix x psi grid is uniform on SO(3)) |
| search points | valid area x `12 nside^2` x `n_psi` x filters x planes; the z is per pixel over all of them, so the same `kappa` treatment applies |

Not modelled: SVD rank truncation and the polar warp (the field is that of the exact
templates), and the gate (the prediction is for the complete search). It needs a complete
HEALPix grid, one Fourier filter and a final stage over all orientations.

## 3. Using it

```yaml
# null.yaml
search: search.yaml          # the search config whose pipeline the nulls repeat
output_dir: nulls            # never the search's own output_dir
mode: both                   # empirical | analytic | both
seeds: [1, 2, 3, 4, 5]       # one phase-randomised copy each
false_positives: [1.0, 5.0]  # targets, per micrograph
border: 16                   # the border your own picks use (default 0, as the export)
# micrographs: [name, ...]   # default: every micrograph of the search config
# peak_floor: 6.0            # default: the final threshold - 1.5
```

```bash
panther-em null null.yaml
```

Outputs in `output_dir`:

* `null_peaks.h5`: per micrograph and copy the distinct null peaks (`y`, `x`, `z`, valid
  area). It is rewritten after every copy; a rerun searches only the copies it lacks (and
  refuses other peak settings or a lower floor unless `overwrite: true`).
* `null_summary.json`, per micrograph: `cuts` (per target: `empirical` with its `complete`
  flag, `analytic`, and the `formula`), `rate` (lambda(t) on a 0.05 grid with Garwood
  68 % intervals), `gate`, `thresholds` (the stages' own), `normalization` (the after-CTF
  report of the source micrograph) and `analytic` (the full prediction: rate, its parts,
  `kappa`, `rho_bar`, neighbour correlations, sampling settings).
* `<name>_s<seed>_result.h5` per copy with `keep_results: true`.

Micrograph `i` of the search config uses seeds `seed + i * seed_stride` (default stride
1000), so every micrograph gets its own copies. Set the search's final threshold to the cut
(`decision.threshold: 7.73`), or pick from the search results' z maps above it.

## 4. Cost

* **Measured:** one full search per copy, plus a few seconds of peak finding: 5 to 6 min
  per 4k copy of the 13-plane production config (dense stage 1) on an RTX 6000 Ada.
* **Analytic:** 21-23 s per 13-plane 4k configuration on one RTX 6000 Ada (3,000
  rotations for the mean template, 64 x 2 x 4 for `kappa`, 128 x 13 for the lattice), plus
  the program's set-up (template and tables, ~30 s). Memory: the padded volume DFT (4.3 GB
  for a 512^3 template) and the 4k image. On 8 CPU threads the same prediction takes 21 min
  (26 GB of memory) and gives the same cut to 1e-7. The predictor's own sampling spread is
  +-0.01 z (seeds 0-2).

## 5. Validation on xenon_131

4k xenon_131, 60S template (B 0, and B-filtered at 60 / 120), the 13-plane production config
(stage 1 48 x 40 fp16 gate on 3 planes, stage 2 local 96 x 80 fp32 on 13 planes 200 A apart,
nside 32 x 256 psi), peaks radius 10, border 16. The measured references are A6's 20
phase-randomised nulls per configuration. Scripts and outputs live outside this repository.

**Analytic null against measured nulls** (1-false-positive threshold):

| Search (13 planes) | Predicted, sampling seed 0 (seeds 1, 2) | Measured: counted / gate-corrected / smoothed | Formula |
|---|---|---|---|
| B 0, after-CTF | **7.781** (7.783, 7.784) | 7.730 / 7.725 / 7.723 | 7.859 |
| B 60, after-CTF | **7.675** (7.680, 7.682) | 7.651 / 7.665 / 7.635 | 7.859 |
| B 120, after-CTF | **7.586** (7.592, 7.596) | 7.569 / 7.580 / 7.522 | 7.859 |
| B 0, before-CTF | **7.895** (7.898, 7.888) | 7.853 / 7.853 / 7.844 | 7.859 |
| B 60, before-CTF | **7.843** (7.848, 7.835) | 7.841 / 7.841 / 7.825 | 7.859 |
| B 120, before-CTF | **7.793** (7.800, 7.785) | 7.792 / 7.792 / 7.872 | 7.859 |

"Gate-corrected" adds the peaks the production gate dropped (from ungated searches of 1-3
nulls); "formula" is the independent-trials cut on the same valid area.

* **Agreement.** Every prediction lies 0.00 to 0.06 z above the measured (gate-corrected)
  cut, so it errs on the safe side: within 0.01 z at B 60 / 120 and 0.04 / 0.06 z at B 0
  (1.4 / 1.6 predicted false positives per micrograph at the measured cut). The CPU
  prototype, built on Leopard-EM's projections, was also 0.04 z high at B 0, so that offset
  belongs to the model, not to this implementation. The predictor's own sampling spread is
  +-0.01 z.
* **Against the formula.** After CTF the formula is 0.13 to 0.28 z too high; the prediction
  recovers 60 % of that gap at B 0 and 95 % or more at B 60 / 120. Before CTF formula and null
  nearly coincide (section 1), and the prediction says so.
* **Ingredients.** `kappa` (4.6 to 9.7e-4), `rho_bar` (0.003 to 0.006), the neighbour
  correlations and, before CTF, the hypothesis-variance spread (5.1 / 6.4 / 7.3 %) agree
  with the prototype. After CTF the model keeps a 1.2 to 1.5 % variance spread (Panther's
  separable `a_n * b_d` leaves about 1 %), which raises its cuts by 0.01 to 0.02 z over a
  model with exactly equal hypotheses.

**Measured null through the runner** (`panther-em null`, B 0, after-CTF, seeds 1-20):

* **Copies.** All 20 are bit-identical to the stored A6 replicates, both as written by
  PICASSO's `phase_randomise` and as read by the search (centred, contrast-inverted).
* **Thresholds.** The counted cut matches the stored A6 nulls at every target:

  | false positives per micrograph | 0.5 | 1 | 2 | 5 | 10 | 20 |
  |---|---|---|---|---|---|---|
  | `panther-em null` | 7.801 | **7.730** | 7.635 | 7.523 | 7.443 | 7.353 |
  | A6 nulls (same replicates) | 7.801 | **7.730** | 7.635 | 7.523 | 7.442 | 7.352 |

  The 1-false-positive cut is 7.7295 in both, and the gate-corrected A6 cut is 7.725.
* **Peaks.** Per copy, 176 to 216 peaks at z >= 7.0. 3,829 of the 3,830 A6 peaks there have
  a runner peak within 1 px with the same z (to 1e-3); the last (z 7.08) has a neighbour
  0.41 z away instead. Pooled counts at z >= 7.0 / 7.3 / 7.5 / 7.7 are 3,866 / 554 / 126 /
  24 against 3,830 / 551 / 125 / 24: this branch's dense stage 1 passes about 1 % more
  near-threshold pixels to stage 2 than the self-seeded stage 1 the A6 runs used. Below the gate the two differ by design: at z >= 6.0, 190k
  against 522k, because A6 also counted the stage-1 z of the pixels the gate dropped.
* **Gate.** The floor is 7.704. The 1-false-positive cut is flagged complete, and the cuts
  for 2 or more false positives are flagged incomplete. Against an ungated search of copy 1,
  the highest null peak the gate drops is 7.561, so in this configuration those flags are
  conservative.
* **Cost.** 296 to 377 s per copy with the dense stage 1 (605 s for the first, which builds
  the kernel), on 3 GPUs at once.

**Synthetic checks** (`tests/test_null.py`, CPU): on a 96 px micrograph with a smooth CTF,
the model's per-(plane, template) noise variance at the grid rows tracks Panther's own noise
model (correlation > 0.99) and after-CTF evens out the planes; with one dense stage and 30
copies, predicted and measured 1- and 3-false-positive cuts agree within 0.15 z.

## 6. Limitations and open points

* **Structure.** A phase-randomised copy keeps the micrograph's amplitudes (CTF,
  astigmatism, ice, detector) but no structure; carbon edges, flat patches or frame-edge
  objects that make false peaks are in neither null. Mask them (`mask`) for the micrograph
  and the nulls alike.
* **The gate.** The measured null is exact for the pipeline as run, but below the gate's
  floor the count is incomplete (flagged). The analytic null predicts the complete search.
* **Analytic model.** 0.00 to 0.06 z above the measured cuts here, highest at B 0 (as for
  the prototype; untraced: rank truncation is the main unmodelled candidate). It assumes a
  dense grid: on a toy with 12 views (nside 1) and a 38 % hypothesis-variance spread it was
  0.26 z low before CTF (fine after). The crowding correction assumes sparse peaks (lambda much
  smaller than area / (pi r^2)); it saturates on tiny images.
* **Self-seeded stages.** The validation runs used a dense stage 1: `seeding` is on
  mdg-seed-local-stage1, not on this branch. Nulls must be run with the same pipeline as the
  data, seeded or not.
* **Sharing nulls between micrographs** is untested beyond micrographs with nearly the same
  spectrum; per-micrograph nulls (the default) are the safe choice.

## 7. Files

| file | change |
|---|---|
| `src/panther_em/managers/null.py` | new: `NullConfig`, `run`; copies, peaks file, summary, gate floor |
| `src/panther_em/search/null_model.py` | new: `predict_null`, `TemplateProjector`, `lag_grid_planes`, `lattice_neighbours`, `reference_grid`, `healpix_nside` |
| `src/panther_em/managers/_common.py` | `read_image`, `centered` (split out of `read_micrograph`, unchanged behaviour) |
| `src/panther_em/managers/search.py` | `_Run.reference_noise`: the after-CTF reference noise kept for the predictor |
| `src/panther_em/cli.py`, `managers/__init__.py` | the `null` program |
| `scripts/threshold_test_pipeline.py` | step 7: nulls of the empty micrograph m1 |
| `tests/test_null.py`, `tests/test_managers.py` | new tests; `null` in the config round trip and CLI tests |
| `docs/null_calibration.md` | this file |
