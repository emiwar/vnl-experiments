# Which behaviours does position-control-without-proprioception actually fail at?

> **Status: Stage A of three.** Everything below rests on the run index alone. The
> per-behaviour decomposition — the question this folder exists for — needs `eval`
> artifacts produced with the new `per_clip` spec key and `trace` artifacts for the 30 s
> clips, both of which need a cluster round-trip. The machinery for all three stages is
> written, tested and committed; [Stages B and C](#stages-b-and-c-what-is-still-missing)
> says exactly what to run. Stage A already moves the headline number, which is why it is
> written up rather than held back.

## Question

[`../efference-copy-vs-proprioception/`](../efference-copy-vs-proprioception/) reports that
with the decoder's proprioception stream ablated, a **position**-actuated policy with a
2-step efference copy reaches 88 % of the proprioception-intact baseline on held-out 5 s
clips, while the torque arm reaches 33 %. Taken at face value that puts the
encoder–decoder premise in doubt: if handing intentions straight to local joint servos
nearly works, the decoder may not be doing much.

A cross-clip mean is the wrong statistic to settle that on, because the clip set is
behaviourally heterogeneous. Three decompositions of the same runs:

1. **By reward term and termination reason.** Is the deficit a tracking deficit or a
   falling-over deficit, and which of the ten reward terms carries it? *(Stage A — answered
   below.)*
2. **By behaviour.** Is the ablated position policy at the intact ceiling on sedentary
   behaviour and far below it on locomotion and rearing, so that 88 % is a mixture of
   ~100 % and ~70 % rather than a uniform 88 %? *(Stage B.)*
3. **At the moment of failure.** What was the reference animal doing when the episode
   ended — which is not the same question as which clips scored badly. *(Stage C.)*

### The interpretation rule, fixed before the numbers were looked at

`pos_nointent` and `torque_nointent` are in the cohort for a reason. They have
proprioception but no imitation target, so they bound the reward available for merely
standing around in a plausible pose. **If a task-blind policy also scores near the intact
ceiling on the sedentary bins, then "near-ceiling on sedentary behaviour" is a fact about
the reward function and not about the policy.** That rule is recorded in
[`extract.py`](extract.py)'s docstring and was committed before Stage A ran. It turned out
to matter more than expected — see the conclusion.

## Dataset & comparability

- **Source:** WandB `emiwar-team/nnx-ppo-rodent-delays`, selected by the `CONDITIONS` in
  [`extract.py`](extract.py) and frozen in [`runs.csv`](runs.csv). **17 runs, 9 conditions,
  all usable** — every run has all three datasets.
- **Metric.** The inline end-of-training eval (`summary.final_eval/<dataset>/*`), which
  carries per-reason `termination_rate`, all ten `reward_terms` and six `errors` for
  `train` / `old_eval` / `new_eval`. One run (`vo7jiiwr`) died in that eval and is read
  from its pinned `eval:eval3ds-382e9e69` artifact instead; its `reward_source` column says
  so and it is excluded from every primary claim.
- **Conditions.** All `AbsoluteImitation`, `rodent_no_tail_collisions.xml`,
  `body_target_frame=reference_root`, `RodentEncDecDelays` (enc/dec `[512]×4`, critic
  `[1024,1024]`), `latent_size=32`, `kl_weight=0.001`, `seed=42`, all twelve PPO
  hyperparameters matched, all trained to 600 M steps.

  | condition | n | decoder inputs | role |
  |---|---|---|---|
  | `pos_noproprio_eff2` | 2 | intention + efference 2 | **the subject** |
  | `pos_intact` | 2 | all three | position ceiling |
  | `torque_intact` | 3 | all three | torque ceiling |
  | `torque_delay10` | 3 | all three, proprioception delayed 10 | **reward-matched control** |
  | `pos_noproprio_eff0` | 3 | intention only | ablation floor |
  | `torque_noproprio_eff2` | 1 | intention + efference 2 | same ablation, other actuator |
  | `pos_nointent` | 1 | proprioception + efference | task-blind floor |
  | `torque_nointent` | 1 | proprioception + efference | task-blind floor |
  | `torque_delay10_aug11` | 1 | all three, delayed 10 | cross-check only |

- **The reward-matched control is well matched.** `torque_delay10` scores 1712 on
  `old_eval` against the subject's 1768 (−3.2 %), and the 2026-08-11 cross-check run 1778
  (+0.5 %). So "same mean reward, different mechanism" is a measured premise, not an
  assumption.

- **Programmatic comparability:** [`comparability.txt`](comparability.txt). Every
  `env_params` entry, every network size and regularisation setting, all twelve PPO
  hyperparameters, `total_steps` and `summary._step` are single-valued across all 17 runs.
  Four environment invariants vary, all expected and none experimental: `git_commit` (six
  values), `gpu` (five models), `cuda_version` and `os` (a cluster upgrade), and
  `repos.nnx_ppo.commit` (two values).

- **Two checks that are not ceremony.**
  - *The reward terms reconstruct the total.* The env's reward is the plain sum of its
    configured terms, so the ten `reward_terms` must add to `episode_reward` per run and
    dataset. They do, to a **4.1 × 10⁻⁷** maximum relative residual over **48/48** rows.
    This is what says the decomposition below is complete rather than missing a term. An
    earlier version of this check reported "48 rows" while silently validating three,
    because the inline summary elides the `/mean` leaf for `reward_terms` and `errors`
    where the artifact keeps it; the coverage count in the verdict line exists because of
    that.
  - *The clip axis is identical across runs.* `_pairable` in [`extract.py`](extract.py)
    gates on `reference_data_path`, `clip_length == 250` and a null `keep_clips_idx`,
    because `ReferenceClips.split()` is `RandomState(0).permutation` over those — if any
    differed, clip *i* would be a different clip for different runs and every paired
    figure in Stage B would be silently wrong. All 17 runs match.

### Caveats

1. **Five of nine cells are n = 1**, including both task-blind floors, which the headline
   rescaling in figure 5B depends on. The floor is the least-replicated and most
   load-bearing measurement here.
2. **One seed.** Every run is `seed = 42`, so nothing here bounds seed-to-seed variation.
   The parent folder measures a 2.51 % run-to-run floor on this cohort's reward at this
   budget; read every difference below against that.
3. **`vo7jiiwr` is selected against a weaker record.** It predates
   `net_params.network_class` / `efference_length` / `dec_use_*` being logged, so the only
   thing separating it from the `RodentForwardModel` run at the same delay in the same
   launch is the `EncDec` tag — a documented departure from "tags are not evidence",
   justified because the tag is the only surviving record. It is kept in its own condition
   and excluded from every primary figure.
4. **`train` is the policy's own training split** and is shown for power, not for
   generalisation. It agrees with `old_eval` throughout.
5. **The `new_eval` numbers are a single unwindowed evaluation** on 32 clips where one
   early termination is expensive, so read its trends and not its point values.
6. **Nothing here is per-behaviour yet.** The conclusions below are about *where in the
   reward function* the deficit sits, not about which behaviours it sits in.

## Figures

![Episode reward per condition, one panel per eval dataset with independent y-axes; one marker per run and a bar at the mean.](figures/aggregate_reward.png)

**Figure 1 — the headline, reproduced from the index alone.** Three panels rather than one
axis or a ratio: `new_eval` clips are 30 s against 5 s, so its reward is ~6× larger before
anything about the policy enters. The subject sits just below the reward-matched torque
control on all three.

![Reward per alive step, mean time alive, and failure hazard, per condition and dataset.](figures/reward_per_alive_step.png)

**Figure 2 — episode reward split into tracking quality and survival.** The two disagree,
which is the first informative thing. On `old_eval` the subject has *lower* per-step
tracking than the reward-matched torque control (3.96 vs 4.07) but lives *longer* (4.47 s
vs 4.20 s) and fails at **0.035/s against 0.058/s** — it reaches the same reward by
surviving better while imitating slightly worse. On `new_eval` that reverses: the torque
control lives longer (15.1 s vs 13.6 s). So the position servo's advantage is a
short-horizon one.

![Stacked per-reason termination fractions per condition, one panel per dataset.](figures/aggregate_failure_modes.png)

**Figure 3 — how episodes ended.** Both intact arms essentially never fail on 5 s clips
(survival 0.99–1.00). `root_too_far` dominates every ablated condition on `old_eval`. On
`new_eval` the subject's failures split almost evenly between `root_too_far` (0.47) and
`root_too_rotated` (0.34), while the ablation floor is overwhelmingly `root_too_far`
(0.69) — the efference copy converts "falls behind the target" into "loses its
orientation", it does not remove the failure.

![Each active reward term per alive step, one panel per term.](figures/reward_terms.png)

**Figure 4 — where the reward comes from.** Per alive step, because the stored terms are
episode totals and would otherwise mostly re-measure lifespan. This is the panel that
reframes the question. `torso_z_range` is **~1.00 for every condition including the
task-blind ones**, and `root_pos` / `root_quat` are ~0.70 there against ~0.98 intact. The
only terms that separate conditions are `joints` and `end_eff`, and `end_eff` does most of
the work: intact 0.84, subject 0.64, task-blind 0.25, torque-ablated 0.16.

![Raw per-step reward, and per-step reward rescaled onto each actuator's own task-blind floor and intact ceiling.](figures/reward_above_floor.png)

**Figure 5 — the headline, corrected for free reward.** Panel A is raw. Panel B is the
folder's one derived axis and is justified by the pre-registered rule above: roughly 70 %
of this reward function is collectable without knowing the imitation target at all, so
"88 % of intact" overstates how much imitating is happening. Rescaled onto each actuator's
**own** floor and ceiling, the subject earns **73 %** of the imitation-attributable reward
on `old_eval`, not 88 %.

## Tentative conclusion (Stage A)

**The 88 % is smaller than it looks, and the reward-matched torque control reaches its
score by almost exactly the same mechanism — but neither of those is yet an answer about
behaviour.**

- **Most of this reward is free.** A policy with proprioception and *no imitation target*
  scores 3.01 of the intact 4.31 reward per alive step on `old_eval` — 70 %. It collects
  `torso_z_range` in full and ~70 % of root position and orientation. Only `joints` and
  `end_eff` discriminate.
- **So the headline drops from 88 % to 73 %.** Measured on per-step reward against the
  same actuator's task-blind floor and intact ceiling, `pos_noproprio_eff2` earns 73.4 %
  of the imitation-attributable reward on `old_eval` (82.8 % on `new_eval`), and
  `pos_noproprio_eff0` 31.1 %. The parent folder's conclusion survives in direction and
  shrinks in size.
- **The deficit is concentrated in the end effectors.** Per alive step the subject keeps
  92 % of intact's `joints` reward but only 76 % of its `end_eff`, and its end-effector
  tracking error is 2.1× intact's (0.040 m vs 0.019 m) while its root position error is
  only 2× (0.012 m vs 0.006 m) on a term that is nearly saturated anyway. The paws are what
  is lost, which is what the mechanism in the parent folder predicts: a commanded joint
  angle is a good proxy for a joint angle and a poor one for where a foot ends up.
- **`torque_noproprio_eff2` is worse than knowing nothing.** At −51 % on the rescaled axis
  it sits *below* its own actuator's task-blind floor in per-step tracking (2.69 vs 3.28).
  The parent folder called this "suggestive, not shown" against a cross-actuator reference;
  with each actuator's own floor measured it is unambiguous, though still n = 1 per side.
- **The reward-matched control is matched in mechanism too, on 5 s clips.** The subject and
  `torque_delay10` agree to within a percent on `joints` (0.588 vs 0.588), `end_eff` (0.639
  vs 0.627) and joint error (1.465 vs 1.452). They differ in survival and in control cost:
  the position policy pays **6× more** (−0.123 vs −0.019 per step). Two very different
  routes to the same actuator command converge on nearly the same tracking.
- **What Stage A cannot say.** Whether any of this is behaviour-selective. A uniform 73 %
  and a mixture of 100 % on grooming with 55 % on locomotion are indistinguishable in every
  figure above, and they imply very different things about what the servos are doing.

## Stages B and C: what is still missing

Both need the cluster, because no run in this cohort has a local checkpoint.

**Stage B — per-clip.** `EvalProducer` gained a `per_clip` spec key (default `None`, so
`eval3ds-382e9e69` and the 393 stored artifacts are untouched and `VERSION` stays at 3 —
the `action_noise` precedent from `analysis/README.md` §2). Set, it keeps the `[n_clips]`
vectors the eval already computes, plus each clip's behaviour label as stamped by the
loader that ran it.

```bash
python -m vnl_experiments.artifacts plan --kind eval \
    --runs analysis/rodent/per-behaviour-failure-modes/runs.csv \
    --set per_clip=true --out todo_per_clip.txt
scp todo_per_clip.txt cannon:$SCRATCH/vnl-experiments/
# on the cluster -- pin one GPU model: per-clip quantities are the worst hit by
# eval nondeterminism, and `plan --out` writes only ids so `--set` must be repeated
sbatch -p gpu_h200 slurm_eval.sh todo_per_clip.txt eval --set per_clip=true
python -m vnl_experiments.artifacts pull --kind eval \
    --runs analysis/rodent/per-behaviour-failure-modes/runs.csv     # needs 2FA
```

**Stage C — per-step traces.** A new artifact kind, `trace` (`VERSION = 1`, HDF5, one per
(run, dataset), 3–10 MB). It records every env metric per step plus `reward`, `alive` and
`running`, including `current_frame` — so the reference frame a step was tracking is *read*
rather than inferred. This is what makes behaviour-resolved reward possible inside a 30 s
clip, which matters because those clips average 13 MotionMapper behaviours each.

```bash
python -m vnl_experiments.artifacts plan --kind trace \
    --runs analysis/rodent/per-behaviour-failure-modes/runs.csv --out todo_trace.txt
sbatch -p gpu_h200 slurm_eval.sh todo_trace.txt trace          # on the cluster
python -m vnl_experiments.artifacts pull --kind trace \
    --runs analysis/rodent/per-behaviour-failure-modes/runs.csv
```

`extract.py` detects both and writes `clips.csv`, `behaviour.csv`, `behaviour_frames.csv`
and `failure_context.csv`; `REQUIRES` grows with them, so `coverage.txt` reads as a plan
until then and as a guarantee afterwards. Four guards are in place and tested: the clip
axis must be identical across runs, the artifact's stamped `clip_names` must equal the
committed labels, the per-clip means must match the independent inline numbers to 8 %, and
a laptop-produced artifact is refused outright.

### Behaviour labels

[`behaviour_labels.py`](behaviour_labels.py) builds both labellings and asserts them;
[`behaviour_labels.txt`](behaviour_labels.txt) is the verdict of every check. They are
**different partitions** and must be compared only through the coarse grouping:

| | `train` / `old_eval` (5 s) | `new_eval` (30 s) |
|---|---|---|
| source | `snips_order` in `reference_clips.h5` | `behavior/motion_mapper` in `2020_12_22_1.h5` |
| granularity | one label per clip | one label per frame |
| classes | 6 | 16 names over 100 clusters |
| `old_eval` / frame mix | FastWalk 37, Rear 35, Walk 33, FaceGroom 27, RGroom 20, LGroom 17 | still 46 %, locomote 37 %, rear 16 %, **groom 2 %** |

The two are complementary rather than redundant: the snippet set has **no still class at
all** and 64 of 169 held-out clips are grooming, while `new_eval` is 46 % still and only
2 % grooming. So the grooming arm of the hypothesis is testable on `old_eval` and the
stillness arm on `new_eval`, and agreement between them would be real replication.

Two traps in the `new_eval` join, both asserted rather than assumed: the MotionMapper ids
are **1-based** (`names[k-1]`; the 0-based reading makes WalkFast slower than ProneStill),
and the frame alignment must be checked on **`pose/keypoints`, never `pose/qpos`** — the
session file is a different STAC fit whose `qpos` differs by up to 1.7 while its keypoints
match the eval clips' source to float32 rounding.

## Follow-ups

1. **Run Stages B and C.** Everything else here is downstream of that.
2. **A second task-blind run per actuator.** Both floors are n = 1 and figure 5B's rescaling
   divides by them, so they are now the weakest link in the headline number.
3. **A second eval seed for the per-clip artifacts** (`--set per_clip=true --set seed=1`)
   would turn the per-clip noise caveat into a measured flip count. Deferred deliberately.
4. **Probe the end-effector claim directly.** The deficit is concentrated in `end_eff`, and
   the per-clip block already carries all 18 per-body errors — enough to say whether it is
   the forelimbs, the hindlimbs or the head, without any new compute once Stage B lands.
5. **The session file also carries `ephys/spike_counts`** (360 000 × 131 units) on the same
   frame index as the behaviour labels. Nothing in this project uses it yet.

---

*Reproduce:* `../.venv/bin/python analysis/rodent/per-behaviour-failure-modes/behaviour_labels.py --verify && ../.venv/bin/python analysis/rodent/per-behaviour-failure-modes/extract.py && ../.venv/bin/python analysis/rodent/per-behaviour-failure-modes/plot.py`
(add `--sync --refresh` to the extract to pull in runs added since `runs.csv` was frozen).
Drift checks: `behaviour_labels.py --check` and `extract.py --check`, both non-zero on drift.
