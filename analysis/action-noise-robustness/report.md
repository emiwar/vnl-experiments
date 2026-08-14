# Is the explicit forward model more sensitive to motor noise than the enc-dec?

> **Status: mechanism in place, sweep not yet run.** `runs.csv` is frozen at 56 runs and
> `coverage.txt` shows 0/280 eval artifacts — all 56 need the cluster (no local
> checkpoints). The smoke-test numbers at the bottom are from 2 runs and 4 clips on a
> *different* cohort; they are a mechanism check, not a result.

## Question

The explicit forward model hands its decoder the predictor's output *instead of* the
delayed proprioception, with no skip path (`vnl_experiments/delays/forward_model.py`).
Does its advantage over the enc-dec therefore depend on execution being clean? An answer
is a curve of performance vs. a fixed action-noise σ for each arm: if the explicit arm's
*fractional* degradation is steeper, the advantage is conditional on clean execution.

Secondary: **does training with wider exploration buy robustness?** The `min_std = 0.25`
tranche has 2.5× the exploration noise of the rest. If robustness is partly just "was
trained under noise", those runs should degrade more slowly.

## Design

Each checkpoint is re-evaluated with a fixed Gaussian perturbation added to the executed
action — σ in post-tanh action units (a fraction of the actuator half-range), clipped
back into `[-1, 1]`.

- **Fixed σ, not the policy's learned `std`.** The learned std is state-dependent and
  differs per run and architecture; reusing it would confound "robust to perturbation"
  with "has a wide learned distribution".
- **Unobserved motor noise.** The noise is added after the network's action and outside
  `EfferenceCopy`, so the efference queue holds the *intended* action while the body
  executes the perturbed one. The predictor's error is then irreducible, not merely
  out-of-distribution. Both arms carry an efference copy and both get a clean queue, so
  the comparison is symmetric. The noise does reach the policy indirectly, `delay_k`
  steps later, via the env's `prev_action` / `actuator_ctrl` channels.
- **σ = 0 is measured, not assumed** — it is a sweep point produced by the same code
  path, spec and batch as the noisy points, rather than the pre-existing
  `eval3ds-66aaff5b` records.
- **σ ∈ {0, 0.02, 0.05, 0.1, 0.25}.** The smoke test below suggests 0.02–0.05 is where the
  graded difference between the *architectures* lives, and that `min_std = 0.1` policies
  are near the floor by 0.1. 0.25 is included regardless, because that saturation was
  measured on `min_std = 0.1` policies and whether the `min_std = 0.25` tranche still has
  headroom at 0.25 is exactly the secondary question. Expect the `min_std = 0.1` arms to
  bottom out there — that is the contrast, not a wasted cell.

## Dataset & comparability

- **Source:** WandB `emiwar-team/nnx-ppo-rodent-delays`, selected by `CONDITIONS` in
  `extract.py` and frozen in `runs.csv`. 56 runs. All new XML
  (`rodent_no_tail_collisions.xml`), `reference_root` targets, torque actuators,
  seed 42, 600 M steps, `[512]×4` encoder/decoder, `[1024]×2` critic.

| condition | n | arm | min_std | delays | commit | state |
|---|---:|---|---|---|---|---|
| `expfm` | 23 | explicit FM | 0.1 | 0–10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100 | `ef060b73` | failed |
| `encdec` | 23 | enc-dec + efference | 0.1 | same 23 | `ef060b73` | failed |
| `expfm_std25` | 5 | explicit FM | 0.25 | 0, 5, 10, 20, 50 | `d02b854a` | finished |
| `pgfm_std25` | 5 | policy-gradient FM | 0.25 | 0, 5, 10, 20, 50 | `d02b854a` | finished |

- **The two primary arms are exactly matched**: same 23 delays, one run per delay per arm,
  same launch tranche and commit. The delay axis is the replication (n = 1 per cell), so
  the comparison is *paired* across 23 delays rather than averaged over seeds.
- **Included despite `state = failed`:** both primary conditions are the 2026-08-11
  `ef060b73` tranche, which died in the *post-training* evaluation. The inclusion rule
  (following `analysis/collision-model-xml`) is `state ∈ {finished, failed}` **and**
  `summary._step == 600_064_000`; a run that died during training cannot satisfy the
  second condition. The three `crashed` forward-model runs at `25732c42` have
  `_step = NaN` and are excluded by that gate.
- **Artifacts used:** `REQUIRES` names one `eval` spec id per σ, each covering all three
  datasets. See `coverage.txt`.
- **Programmatic comparability:** `comparability.txt` — every condition is single-valued
  on every invariant, including `git_commit`. Pooled across conditions only `min_std` and
  `git_commit` vary, and they vary together: the `std25` tranche is a separate launch.
- **Manual comparability:** *(to do before reporting)* `git diff ef060b73 d02b854a` —
  whether anything between the two tranches touches shared env / reward / network code.
  Only matters for the secondary (`min_std`) comparison; the primary pair is one commit.
- **Caveats:**
  - `min_std = 0.25` exists **only** for the forward model — there is no enc-dec run at
    0.25 anywhere in the project. So the exploration-width axis cannot be crossed with the
    architecture comparison, and `pgfm_std25` has no `min_std = 0.1` twin inside this
    cohort (`pgfm_new_reference`, 13 runs at `25732c42`, would be the twin if the budget
    allows adding it).
  - n = 1 per (condition, delay). MuJoCo Warp GPU physics is not bit-reproducible:
    `episode_reward` moves ~1 % and `termination_rate` ~3 pp per flipped clip
    (README §6). The per-run σ=0 normalisation in `plot.py` cancels most of that, since
    numerator and denominator come from the same checkpoint.
  - Each (run, σ) is a single *noise realisation*. If the spread across delays turns out
    comparable to the between-arm difference, repeat with several `seed` values per σ
    (a distinct spec field, hence a distinct `spec_id`).
  - `train` here is the 80 % training split, so the `train` vs `old_eval`/`new_eval`
    contrast is what says whether noise sensitivity is a generalisation phenomenon.

## Producing the sweep

280 evaluations (56 runs × 5 σ), all three datasets each, all needing the cluster.

```bash
# laptop: one plan per sigma
for s in 0.0 0.02 0.05 0.1 0.25; do
  python -m vnl_experiments.artifacts plan --kind eval \
      --runs analysis/action-noise-robustness/runs.csv \
      --set action_noise=$s --out todo_n$s.txt
done

# cluster
sbatch slurm_eval.sh todo_n0.02.txt eval --set action_noise=0.02   # etc.

# laptop
python -m vnl_experiments.artifacts pull --kind eval \
    --runs analysis/action-noise-robustness/runs.csv
../.venv/bin/python analysis/action-noise-robustness/extract.py --refresh
../.venv/bin/python analysis/action-noise-robustness/plot.py
```

σ = 0.0 is a real spec point (`eval3ds-n00-04ceda93`), distinct from the noise-free
`eval3ds-66aaff5b`; produce it like the others.

## Smoke test (not a result)

Two delay-20 runs from the **old-XML** cohort — `bge1cw3s` (explicit FM) and `fku7oyos`
(enc-dec) — on `old_eval` with `limit_clips=4`. Different cohort, 4 clips, one run each:
its only purpose was to check the mechanism end to end.

| arm | σ | reward/step | lifespan (steps) | survived | fm_pred_mse |
|---|---|---|---|---|---|
| enc-dec | 0.00 | 4.08 | 227 | 0.25 | — |
| enc-dec | 0.10 | 2.93 | 46 | 0.00 | — |
| enc-dec | 0.25 | 1.95 | 24 | 0.00 | — |
| explicit FM | 0.00 | 4.06 | 318 | 0.25 | 0.034 |
| explicit FM | 0.10 | 3.09 | 30 | 0.00 | 0.627 |
| explicit FM | 0.25 | 2.17 | 18 | 0.00 | 1.629 |

Two things it settled:

1. **The perturbation reaches the quantity it was meant to reach.** `fm_pred_mse` rises
   18× from σ = 0 to σ = 0.1 — the predictor really is being driven off its training
   distribution, which is the mechanism the question is about.
2. **σ = 0.1 is already near-saturating for `min_std = 0.1` policies.** Both arms lose
   ~90 % of lifespan there, so the architecture comparison should be decided by 0.02 and
   0.05. This is *not* a reason to drop 0.25: these two checkpoints were trained with
   `min_std = 0.1`, and the `min_std = 0.25` tranche may well still have a floor left at
   σ = 0.25. If even 0.02 turns out to bite hard, add levels *below* it (0.005 / 0.01)
   rather than trusting interpolation from 0.

## Tentative conclusion

None yet.

## Follow-ups

- Add `pgfm_new_reference` (13 runs, `25732c42`, min_std 0.1) to give `pgfm_std25` a
  matched-`min_std` twin — +52 evals.
- Repeat at several noise realisations per (run, σ) if the across-delay spread is large
  relative to the arm difference.
- The mirror experiment: inject the noise *inside* the sampler so the efference queue
  holds the executed action. That separates "the noise is unpredictable" from "the noise
  shifts the predictor's input distribution". Requires a change in `nnx-ppo`.
- Sensory (proprioception) noise as a second axis, which stresses the encoder rather than
  the predictor.

---

*Reproduce:* `../.venv/bin/python analysis/action-noise-robustness/extract.py && ../.venv/bin/python analysis/action-noise-robustness/plot.py`
(add `--sync --refresh` to the extract to pull in runs added since `runs.csv` was frozen).
