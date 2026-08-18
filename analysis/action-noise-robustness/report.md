# Is the explicit forward model more sensitive to motor noise than the enc-dec?

**Yes — and the effect appears exactly where the mechanism predicts it should.** The
explicit forward model starts ~8 % *ahead* of the enc-dec with no noise and is ~10 %
*behind* it at σ = 0.02: its advantage is conditional on clean execution. The penalty
scales with the delay the predictor has to bridge (Spearman ρ ≈ −0.9, p < 0.001) and
vanishes at delay 0, where there is nothing to predict.

A second, **larger** effect fell out of the same sweep: training with wider exploration
(`min_std = 0.25`) costs ~1/3 of the noise-free reward but buys far more robustness than
either architecture choice, overtaking the `min_std = 0.1` runs by σ ≈ 0.05 and beating
them 4× at σ = 0.25.

## Question

The explicit forward model hands its decoder the predictor's output *instead of* the
delayed proprioception, with no skip path (`vnl_experiments/delays/forward_model.py`).
Does its advantage over the enc-dec depend on execution being clean? An answer is a curve
of performance vs. a fixed action-noise σ for each arm: if the explicit arm's *fractional*
degradation is steeper, the advantage is conditional.

## Design

Each checkpoint was re-evaluated with a fixed Gaussian perturbation added to the executed
action — σ in post-tanh action units (a fraction of the actuator half-range), clipped back
into `[-1, 1]`. 56 runs × 5 σ × 3 datasets.

- **Fixed σ, not the policy's learned `std`**, which is state-dependent and differs per
  run and architecture; reusing it would confound robustness with distribution width.
- **Unobserved motor noise.** The noise is added after the network's action and outside
  `EfferenceCopy`, so the efference queue holds the *intended* action while the body
  executes the perturbed one. The predictor's error is irreducible, not merely
  out-of-distribution. Both arms carry an efference copy and both get a clean queue.
- **σ = 0 is measured**, from the same code path, spec and batch as the noisy points.
- **Headline metric is cumulative `episode_reward`**, which folds in tracking quality and
  survival, compared only *within* a dataset. Each run is normalised to its **own** σ = 0
  value, which also cancels the ~1 % GPU-physics irreproducibility (numerator and
  denominator come from the same checkpoint). `reward_per_step` conditions on being alive
  and is the rate-only view.
- **Paired by delay.** With n = 1 per (condition, delay), the 23 matched delays *are* the
  replication, so the primary test is a paired one across delays, not a seed average.

## Dataset & comparability

WandB `emiwar-team/nnx-ppo-rodent-delays`, selected by `CONDITIONS` in `extract.py`,
frozen in `runs.csv`. All new XML (`rodent_no_tail_collisions.xml`), `reference_root`
targets, torque actuators, seed 42, 600 M steps, `[512]×4` enc/dec, `[1024]×2` critic.

| condition | n | arm | min_std | delays | commit | state |
|---|---:|---|---|---|---|---|
| `expfm` | 23 | explicit FM | 0.1 | 0–10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100 | `ef060b73` | failed |
| `encdec` | 23 | enc-dec + efference | 0.1 | same 23 | `ef060b73` | failed |
| `expfm_std25` | 5 | explicit FM | 0.25 | 0, 5, 10, 20, 50 | `d02b854a` | finished |
| `pgfm_std25` | 5 | policy-gradient FM | 0.25 | 0, 5, 10, 20, 50 | `d02b854a` | finished |

- **The primary arms are exactly matched**: same 23 delays, one run per delay, same launch
  tranche and commit.
- **Included despite `state = failed`:** both primary conditions are the 2026-08-11
  `ef060b73` tranche, which died in the *post-training* evaluation. The rule (following
  `collision-model-xml`) is `state ∈ {finished, failed}` **and** `summary._step ==
  600_064_000`; a run that died during training cannot satisfy the second. The three
  `crashed` runs at `25732c42` have `_step = NaN` and are excluded by that gate.
- **Coverage: 801/840 cells.** The only gap is **`encdec` at σ = 0.02, 10/23 delays**
  (present: 1, 2, 3, 9, 10, 12, 15, 30, 60, 80). Every other (condition, σ) cell is
  complete. All pooled comparisons in the figures are restricted to delays both arms
  share at that σ (`paired_delays` in `plot.py`), so no panel compares differently
  composed cohorts.
- **Programmatic comparability:** `comparability.txt` — every condition is single-valued
  on every invariant, `git_commit` included. Pooled, only `min_std` and `git_commit` vary,
  and they vary together (the `std25` tranche is a separate launch 2 days later).
- **Manual comparability:** *(outstanding)* `git diff ef060b73 d02b854a` — needed only for
  the secondary `min_std` comparison; the primary pair is a single commit.

## Result 1 — the explicit forward model degrades faster

`old_eval`, paired by delay, episode reward relative to each run's own σ = 0:

| σ | paired delays | expfm | encdec | difference | expfm worse in | Wilcoxon |
|---|---:|---:|---:|---:|---:|---:|
| 0.02 | 10 | 0.678 | 0.855 | **−0.177** | 7/10 | p = 0.027 |
| 0.05 | 23 | 0.316 | 0.400 | **−0.084** | 19/23 | p < 0.001 |
| 0.10 | 23 | 0.139 | 0.168 | −0.029 | 21/23 | p < 0.001 |
| 0.25 | 23 | 0.037 | 0.046 | −0.009 | 18/23 | p < 0.001 |

The direction is essentially unanimous (18–21 of 23 delays at every σ). In absolute terms
the ranking **reverses** between σ = 0 and σ = 0.02. Restricted to the 10 delays both arms
share at σ = 0.02, so every row compares the same runs (`old_eval`):

| σ | expfm | encdec | expfm / encdec | survived (expfm / encdec) |
|---|---:|---:|---:|---|
| 0.00 | **1383** | 1285 | 1.08 | 0.55 / 0.48 |
| 0.02 | 1011 | **1128** | **0.90** | 0.35 / 0.40 |
| 0.05 | 584 | **618** | 0.94 | 0.19 / 0.18 |
| 0.10 | 158 | **168** | 0.94 | 0.01 / 0.01 |

Over the full 23 delays (valid at every σ except 0.02) the ratio is 1.09 at σ = 0 and
0.93 / 0.95 / 0.99 at σ = 0.05 / 0.10 / 0.25 — same reversal, and it narrows as both arms
approach the floor.

![Degradation](figures/degradation.png)

Absolute (top) and relative-to-own-σ=0 (bottom) episode reward, per dataset. The two arms
sit on top of each other in absolute terms from σ = 0.05 down; the bottom row is where the
consistent separation lives.

## Result 2 — the penalty tracks the delay, and disappears at delay 0

![Gap vs delay](figures/gap_vs_delay.png)

The explicit arm's paired disadvantage per delay. It is ~0 for delays 0–5, then grows
monotonically: at σ = 0.05 it reaches −0.14 to −0.26 by delays 50–100.

| σ | Spearman ρ (delay, gap) | p |
|---|---:|---:|
| 0.02 | −0.67 | 0.03 |
| 0.05 | −0.92 | < 0.001 |
| 0.10 | −0.87 | < 0.001 |
| 0.25 | −0.91 | < 0.001 |

This is the analysis's strongest internal control. At delay 0 the predictor has nothing to
bridge, so a mechanism that runs *through* prediction error must show no penalty there —
and it doesn't. A generic "this architecture is just more fragile" story predicts no such
delay dependence.

The mechanism is directly visible: the predictor's own L2 error against true current
proprioception rises **25×** across the sweep (`expfm`, shared delays, `old_eval`):
0.054 → 0.094 → 0.216 → 0.449 → 1.364.

![Prediction error](figures/prediction_error.png)

## Result 3 — it is not a generalisation effect

Relative episode reward at σ = 0.05 is 0.304 / 0.387 (expfm / encdec) on **train**,
0.316 / 0.400 on `old_eval`, 0.234 / 0.353 on `new_eval`. The same ordering and roughly
the same magnitude appear on the training clips, so this is a control phenomenon, not
overfitting. It is somewhat *larger* on `new_eval`, i.e. noise and novel clips compound.

## Result 4 — wider training exploration buys much more robustness than architecture does

Shared delays (0, 5, 10, 20, 50), `old_eval`, absolute episode reward:

| σ | encdec | expfm | expfm_std25 | pgfm_std25 |
|---|---:|---:|---:|---:|
| 0.00 | 1373 | **1434** | 934 | 968 |
| 0.02 | **1173** | 1066 | 907 | 955 |
| 0.05 | 655 | 615 | 809 | **897** |
| 0.10 | 479 | 461 | 614 | **695** |
| 0.25 | 72 | 71 | 265 | **305** |

![Exploration width](figures/exploration_width.png)

`min_std = 0.25` costs ~1/3 of the noise-free reward, crosses over between σ = 0.02 and
0.05, and by σ = 0.25 retains 19–20 % of its baseline against 5 % — with survival 0.10–0.14
vs **0.000**, i.e. the `min_std = 0.1` policies never finish a clip at all. The effect
dwarfs the architecture difference.

Two details reinforce Result 1. `pgfm_std25` (predictor trained only by the policy
gradient) beats `expfm_std25` at *every* σ — and its baseline prediction error is 0.710 vs
0.074, i.e. it never learned to predict well but also never came to depend on the
prediction being right. The more a policy leans on an accurate forward prediction, the more
unobserved motor noise costs it.

## Caveats

- **σ = 0.02 is the weakest row.** `encdec` has 10/23 delays there, and the present set
  skews mid-to-long (9, 10, 12, 15, 30, 60, 80) — exactly where the gap is largest. The
  −0.177 mean is therefore an *over*-estimate of what a full 23-delay set would give: on
  those same 10 delays the σ = 0.05 gap is −0.079, close to the full-set −0.084, so the
  delay skew alone does not explain why σ = 0.02 shows the largest gap. Every
  number quoted at σ = 0.02 above is paired on those 10 delays, so it is internally
  consistent, but it is 10 delays and not 23. Filling the 13 missing runs (13 evals) would
  settle the most quotable number in the report.
- n = 1 per (condition, delay); the pairing is across delays, and the reported spread is
  across-delay spread, not seed spread.
- One noise realisation per (run, σ). The consistency across 23 independent delays is what
  carries Result 1, not any single cell.
- At σ = 0.25 both `min_std = 0.1` arms are on the floor (survived = 0.000), so the small
  gap there says little; Results 1–2 rest on σ = 0.02–0.10.
- The `min_std` comparison confounds exploration width with launch tranche/commit
  (`ef060b73` vs `d02b854a`, 2 days apart) and rests on 5 delays × 1 run.
- No enc-dec run exists at `min_std = 0.25` anywhere in the project, so Result 4 is
  internal to the forward model.

## Follow-ups

- **Fill `encdec` σ = 0.02** (13 evals) — cheapest way to firm up the headline number.
- Add `pgfm_new_reference` (13 runs, `25732c42`, min_std 0.1) so the policy-gradient arm
  has a matched-`min_std` twin, separating "implicit predictor" from "wide exploration".
- Train an enc-dec at `min_std = 0.25` to cross the two axes.
- The mirror experiment: inject noise *inside* the sampler so the efference queue holds the
  executed action. That separates "the noise is unpredictable" from "the noise shifts the
  predictor's input distribution", and Result 2 predicts the penalty should largely vanish.
- Sensory (proprioception) noise as a second axis, stressing the encoder rather than the
  predictor.

---

*Reproduce:* `../.venv/bin/python analysis/action-noise-robustness/extract.py && ../.venv/bin/python analysis/action-noise-robustness/plot.py`
(add `--sync --refresh` to the extract to pull in runs added since `runs.csv` was frozen).
