# New walker XML + reference_root vs the old baseline

## Question

The configuration going forward is **new (almost-full-collision) XML + `reference_root`
target frame + torque actuators**. How does it compare with the old baseline it replaces
(**old sparse-collision XML + `current_root` + torque**), across observation delays? That
contrast moves two things at once — can the walker XML and the target frame be told apart?
And is position control different?

> **Note on a revision.** An earlier version of this report was built on training-time
> WandB reward, because no new-`reference_root` run had an offline evaluation yet. With
> the offline evals now in (146/149 runs), **several of its conclusions do not survive** —
> in particular "the new XML is free out to delay 20" and "the explicit forward model is
> immune". Training-time reward turns out to under-detect this specific effect, and
> asymmetrically. See *Why the training-time metric missed it*.

## Dataset & comparability

**Source:** WandB `emiwar-team/nnx-ppo-rodent-delays`, tag `TrainEvalSplit`, selected by
the `CONDITIONS` in `extract.py` and frozen in `runs.csv` (149 runs).

| condition | n | XML | control | frame | network | delays | commit |
|---|---|---|---|---|---|---|---|
| `encdec_old_current` | 22 | old | torque | current | EncDec | 0–100 | `1cd5838f` |
| `encdec_old_reference` | 6 | old | torque | reference | EncDec | 0–50 | `909e774d` |
| `encdec_new_current` | 6 | new | torque | current | EncDec | 0–50 | `201d6e11` |
| `encdec_new_reference` | 23 | new | torque | reference | EncDec | 0–100 | `ef060b73` |
| `expfm_old_current` | 23 | old | torque | current | explicit FM | 0–100 | `54643764` |
| `expfm_old_reference` | 6 | old | torque | reference | explicit FM | 0–50 | `909e774d` |
| `expfm_new_reference` | 23 | new | torque | reference | explicit FM | 0–100 | `ef060b73` |
| `pgfm_old_current` | 14 | old | torque | current | PG-FM | 0–100 | `d33e5bcf`, `d4bd4dc0` |
| `pgfm_new_reference` | 13 | new | torque | reference | PG-FM | 0–100 | `25732c42` |
| `expfm_old_position` | 8 | old | position | current | explicit FM | 0–100 | `891cd0d3` |
| `expfm_new_position` | 5 | new | position | current | explicit FM | 0–50 | `201d6e11` |

The primary contrast is measured in **three independent networks**, with full 0–100 delay
sweeps on both sides for EncDec and the explicit FM. The four EncDec cells form a complete
**2 × 2** in XML × frame.

**Primary evidence is the offline evaluation**, not training-time reward: the held-out
`old_eval` split (169 unseen clips, full 502-step rollouts from frame 0), batch spec
`eval3ds-66aaff5b`, 146/149 runs. Only `pgfm_new_reference` has gaps (10/13; missing
delays 5, 20, 30).

**Included despite `state = failed`:** the 46 runs at `ef060b73` (2026-08-11) that make up
`encdec_new_reference` and `expfm_new_reference` died in the *post-training* evaluation.
All reached `summary._step = 600_064_000`, ran a normal 3.5–7.5 h, and have no
`final_eval/*` keys. The inclusion rule is `state ∈ {finished, failed}` **and**
`_step == EXPECTED_STEP`; a run that died during training cannot satisfy the second.

**Excluded by design** (`min_std == 0.1`, `_step == 600_064_000`): the larger-`min_std`
(0.25) and 2 G-step new-XML runs, per request. Also seeds 43/44, non-standard
architectures, `efference_length != delay_k`, and the `fm_loss_weight = 0` **detached**
control.

**Comparability.** Every invariant is single-valued within each condition except
`git_commit` in `pgfm_old_current`. Manual checks: configs read from the index, not tags;
`min_std` / `latent_min_std` / `std_scale` verified (this is what separates in-scope from
out-of-scope new runs and is invisible in a run's name); the `d4bd4dc0`→`d33e5bcf` diff is
additive-only; the `d33e5bcf`→`25732c42` network diff is a default-inert `latent_key`
signature change; and across every primary pair all network, PPO and env invariants are
identical, which is stronger evidence than reading a six-week diff.

**Every eval was verified to have used the run's own body and frame** — `walker_xml`,
`body_target_frame`, `env_class` and the restored checkpoint directory were checked
per run against the run's config, and all 146 restored `checkpoint_step = 600_064_000`. A
body/policy mismatch would produce exactly this report's signature, so this check matters.

**Caveats.**

1. **Single seed per cell** (all seed 42). What carries weight is that the shape
   replicates across three networks, not any single point.
2. **Not commit-controlled**: six to eight weeks separate the arms of each primary pair.
3. `encdec_new_current` and `expfm_new_position` reach only delay 50, limiting the 2 × 2
   and the position contrast to delays ≤ 50.
4. Offline eval is not bit-reproducible (~1 % on reward; `survived` moves by ~0.6 pp per
   clip at 169 clips). Differences below a few percent mean nothing.

## Primary result — the new body falls over far more often

![Held-out performance in three networks](figures/primary.png)

On held-out clips the new configuration is **worse at essentially every delay, in all
three networks**, and the effect decomposes cleanly:

- **Per-step tracking is almost unchanged.** Reward per step moves by −1 % at short delays
  and at worst −11 % at long ones. The policies still imitate about as accurately as before
  while they are upright.
- **Survival collapses.** That is the entire effect.

| delay | survival, EncDec old → new | explicit FM old → new | PG-FM old → new |
|---:|---|---|---|
| 0 | 99 % → 96 % | 99 % → 93 % | 99 % → 91 % |
| 10 | 71 % → 56 % | 88 % → 57 % | 85 % → 62 % |
| 20 | 43 % → 30 % | 57 % → 41 % | — |
| 50 | 26 % → 6 % | 44 % → 12 % | 26 % → 4 % |
| 100 | 7 % → 0 % | 36 % → 17 % | 2 % → 2 % |

Episode reward, which multiplies the two, drops 20–50 % across the middle and upper delay
range in all three networks.

**The explicit forward model is *not* immune** — the earlier training-clip reading of this
was wrong. It is consistently the *best* architecture (44 % vs 26 % survival at delay 50 on
the old body, 12 % vs 6 % on the new; 17 % vs ~0 % at delay 100), and it retains the most
at long delay, but it takes the same qualitative hit. The PG-FM row at delay 100 reads
2 % → 2 % only because its baseline has already hit the floor.

## Why the training-time metric missed it

![Training-time vs held-out](figures/training_vs_heldout.png)

Training-time reward said the new configuration was within ±3 % out to delay 20 and *ahead*
at long delay for the explicit FM. The held-out evaluation says −20 to −50 %. Same runs,
same checkpoints.

The right-hand panel isolates the mechanism. For the **old** body the two lifespan
measurements agree — median ratio 0.96 (EncDec) and 0.97 (explicit FM) at delays ≤ 60. For
the **new** body the training-time metric **overstates** lifespan, by up to 2.3× (EncDec)
and 2.1× (explicit FM), and increasingly so with delay.

So the training-time metric is not merely noisy here; it is **biased in favour of the new
body**. The two evaluations differ in episode structure — training-time reward is measured
on short auto-resetting episodes on the training clips, while the offline eval runs a full
502-step latched rollout from frame 0 — and the new body's failure mode is one that
accumulates over a long continuous rollout. A short window rarely samples it.

This is not a generalization gap: within each arm the train → held-out drop in reward per
step is −0.5 % to −2.5 %, and it is the same for both arms. The offline **`train`**
evaluation (673 clips the policies were trained on) shows the same collapse as `old_eval`.
The difference is rollout length, not clip novelty.

## Decomposition — XML or frame?

![The EncDec 2x2 on held-out clips](figures/decomposition.png)

| delay | XML at `current_root` | XML at `reference_root` | Frame on old XML | Frame on new XML |
|---:|---:|---:|---:|---:|
| 0 | −2.0 % | −2.2 % | −0.8 % | −1.0 % |
| 5 | −7.2 % | −10.8 % | +2.8 % | −1.1 % |
| 10 | −16.8 % | −12.5 % | +0.3 % | +5.4 % |
| 20 | −21.9 % | −30.1 % | +10.7 % | −0.9 % |
| 50 | −48.0 % | −37.2 % | +0.2 % | +20.9 % |

(reward; survival tells the same story — the XML costs 16–19 pp at delay 50 at both frames,
the frame costs −2.4 to +10.7 pp with no trend)

**The XML carries the entire effect**, and does so at both frames. **The frame remains
free** at both XMLs — no trend, no consistent sign, and if anything slightly positive. The
frame conclusion is the one thing that survives unchanged from the training-clip analysis,
and it is now confirmed on held-out data in two networks (`frame_expfm`: −5.2 % to +2.4 %).

## Position control

![Position control](figures/position.png)

The earlier conclusion that position control is unaffected **does not survive** either. On
held-out clips the XML costs position control −4.8 % at delay 0, −10.1 % at delay 20 and
−14.5 % at delay 50, with survival down 6.5–17.2 pp.

It is still clearly *less* affected than torque control — −14.5 % versus −37 to −48 % at
delay 50 — so the qualitative reading (the inner PD loop absorbs part of the extra contact
disturbance) stands. But "free" was an artifact of the training-clip metric.

## Convergence

![Learning curves](figures/convergence.png)

Read these for convergence only; their *levels* are the metric that misleads. At delay 30
the new arm is still climbing hard at 600 M (gaining 9.4 % of its final reward in the last
100 M steps, against 3.2 % for the baseline); at delays 50 and 60 it has flattened or
turned down (−0.4 %, −6.5 %) while the baseline still gains ~3 %. Longer training will
narrow part of the gap but there is no sign it would close it.

## Speed

![Throughput](figures/throughput.png)

Unchanged and settled: 18 A100-matched, delay-matched pairs, all between −0.4 % and
+2.7 %; median ratios 1.005 (explicit FM, n = 14) and 1.014 (EncDec, n = 3). The new XML
is not slower.

## Bottom line

- **`reference_root` is free.** Confirmed on held-out data, at both XML levels, in two
  networks, with no interaction. Adopt it.
- **No speed penalty.** Settled.
- **The new XML has a real and substantial cost**: the body falls over much more often —
  survival at delay 50 drops from 26 % to 6 % (EncDec) and 44 % to 12 % (explicit FM). It
  is a *stability* cost, not a tracking cost: reward per step is nearly unchanged.
- **It affects every network and both control modes**, though the explicit forward model
  and position control are hit less hard.
- **Training-time reward should not be used to judge this change.** It understates the
  cost, and it does so more for the new body than the old.

## Follow-ups

- **This is a stability problem, so look at terminations.** The eval records carry
  per-reason `termination_rate` (`root_too_far`, `root_too_rotated`, `pose_error`). Which
  one grows on the new body would say whether this is falling, drifting, or pose
  divergence — the single most informative next cut, and it needs no new runs.
- **Watch the rendered rollouts.** Videos exist for 8 `pgfm_new_reference` runs; seeing the
  failure directly is likely worth more than another summary statistic.
- **Is it trainable away?** The 2 G-step runs, and possibly a reward/termination-threshold
  adjustment for the more contact-rich body.
- **Multiple seeds**, still n = 1 per cell everywhere.
- **The 3 missing PG-FM evals** (delays 5, 20, 30) sit in the most interesting part of the
  range; worth topping up.
- **A new-XML `current_root` cell for the explicit FM and PG-FM**, to confirm the
  no-interaction result outside EncDec.

---

*Reproduce:* `../.venv/bin/python analysis/collision-model-xml/extract.py && ../.venv/bin/python analysis/collision-model-xml/plot.py`
(add `--sync --refresh` to pull in runs added since `runs.csv` was frozen).
