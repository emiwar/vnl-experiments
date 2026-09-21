# dm_control_suite analyses

Delay experiments on the dm_control tasks, in `nnx-ppo-delays`. The shared pipeline — run
index, artifact store, the question folder, the comparability protocol, the plotting
conventions — is [`../README.md`](../README.md); this file holds what is true of *this
track only*.

These tasks are the cheap, fast-turnaround version of the rodent's question: does a delay
hurt, and can an efference copy, an explicit forward model or recurrence recover it. They
are small enough to sweep nine tasks × several delays × several architectures, which the
rodent is not.

| | |
|---|---|
| WandB project | `nnx-ppo-delays`, pinned by every env group (`conf/env/dmc/*.yaml`) |
| Run index | `../_runs/nnx-ppo-delays.jsonl` |
| Template | `cp -r ../_template/dm_control_suite analysis/dm_control_suite/<question-slug>` |
| Control step | **per task**: 10 / 20 / 25 ms — see the table under Traps, and never the rodent's 10 by default |
| Tasks | `WalkerWalk` `WalkerRun` `WalkerStand` `CheetahRun` `HumanoidWalk` `HumanoidStand` `CartpoleBalance` `CartpoleSwingup` `BallInCup`, plus the small bodies added for the joint-stiffness sweep: `ReacherEasy` `ReacherHard` `HopperStand` `HopperHop` `FingerTurnEasy` `FingerTurnHard`. `envs/registry.py` is the list; this one goes stale |
| Networks | `DelayedMLP`, `FlatForwardModel`, `FlatRecurrent` (flat-obs; the rodent's are dict-obs) |

`train.py` passes no wandb entity, so the full `entity/project` string is whatever your
default entity resolves to — expected to be `emiwar-team/nnx-ppo-delays`, matching the
rodent project's entity. Confirm it once at the first sync, then pin that exact string as
`PROJECT` in each `extract.py`; `--project` has no default, so nothing falls back to the
rodent project behind you.

```bash
python -m vnl_experiments.wandb_utils.index sync --project emiwar-team/nnx-ppo-delays
```

That writes `../_runs/nnx-ppo-delays.jsonl`. It needs network and WandB auth, and is the
one step of this pipeline that cannot be done offline.

Launching runs is `slurm_dmc_delays.sh`, whose header documents the override syntax:

```bash
sbatch slurm_dmc_delays.sh env=dmc/walker_walk net=flat_recurrent train=dmc delay=10 \
    net.rnn_cell=gru
python -m vnl_experiments.sweep --script slurm_dmc_delays.sh \
    env=dmc/cheetah_run net=delayed_mlp train=dmc delay=0,1,2,5,10,20
```

### Preemption and requeue

`slurm_dmc_requeue.sh` is the same thing on `gpu_requeue`, which is nearly free but kills
and requeues a job whenever the node's owner wants it back. Worth it when a run will not
fit a dedicated allocation comfortably — HumanoidWalk at 1 G steps is ~8.8 h on an A100
against a 12 h limit. It takes the same overrides; the trainer keys the run directory and
the WandB run on the Slurm job id, so every attempt continues one curve.

Two things about a resumed curve are worth knowing before reading one, because both are
visible in it and neither is a bug. A light checkpoint (the default) omits the env states,
so on resume:

* **Episode phase is redrawn, not continued.** Each env restarts its episode with the
  phase drawn over the whole episode length — see `EnvSpec.resume_reset` in
  `envs/registry.py`. The truncation rate therefore has a transient that washes out within
  one episode (≤ 1000 steps, against a 480 M–1 G budget).
* **The network carry is zeroed.** The actor's delay and efference queues, and a
  `FlatRecurrent` run's RNN state, start empty, so for the first `delay_k` steps after each
  resume the actor sees zeros where observations should be. At `delay_k ≤ 25` that is a
  ~25-step transient per preemption.

`requeue.full_checkpoints=true` keeps the env states instead and resumes exactly, at the
cost of a much larger save — the env states are ~94 % of the bytes, and for a contact-rich
task at `n_envs = 8192` that has not been measured, so check it fits the preemption grace
period before relying on it.

## There are two eras of run, and they do not share a schema

*(Config **schema** splits at 2026-07-08, below. Two other boundaries cut across it and are
in Traps: the `eval/*` **meaning** change at `971ab99`, 2026-09-09, and the eval/video
**cadence** change of 2026-09-15.)*


90 runs (2026-05-29 → 2026-07-08, delays 0–25) were trained by the pre-Hydra
`train_delays.py`, which was deleted in the 2026-09-01 migration. Everything since comes
from `vnl_experiments/train.py` with `train=dmc`. `conf/train/dmc.yaml` is a faithful
transcription of the old script's PPO settings — asserted field by field in
`vnl_experiments/config/dmc_equivalence_test.py`, which is what makes the two eras
*optimisation*-comparable at all. What is **not** shared is everything around that:

| | legacy (≤ 2026-07-08) | current (Hydra) |
|---|---|---|
| wandb config | flat: `env`, `network`, `delay_k`, `efference_length`, `n_actor_params`, `seed`, `config` (ppo only) | `env_params.*`, `net_params.*`, `config.ppo.*`, `repos.*` |
| `config.json` | not written | written, so the run reloads offline |
| run name | `{Env}_{MLP\|FM}_delay{d}_eff{e}` | `{Task}_{DelayedMLP\|FlatForwardModel\|DelayedLSTM…}_delay{d}_eff{e}` |
| tags | `(env, "DelayedMLP"\|"ForwardModel", delay{d}, eff{e})` | + `network_class`, `FlatObs`, architecture family |

**Consequences for `extract.py`:**

* **A legacy run has no `env_params.*` and no `net_params.*` at all.** Selecting on them
  silently returns zero legacy runs — the same failure mode as the rodent's
  `dec_use_intention` trap, where the column is *absent* rather than `False`. Select the
  task on `config.env` for legacy and `env_spec.task` (or the tag) for current, and
  coalesce with `pipeline.first_present`. The same for `delay_k`: top-level in legacy,
  `net_params.delay_k` now.
* **The network token in the run name changed**, so a cohort selected by name substring
  will silently be one era only. `FM` → `FlatForwardModel` and `MLP` → `DelayedMLP`; the
  legacy *tag* `DelayedMLP` happens to match the current `network_class`, but the legacy
  forward-model tag is `ForwardModel`, which is the **rodent** architecture's name.
* Legacy runs log `n_actor_params`; current runs record full `param_counts` in their eval
  artifacts instead.

## Traps

The general ones are in [`../README.md` §6](../README.md#6-traps) and apply here in full —
particularly the nondeterminism ones, since these tasks are also MuJoCo. Below are the
ones peculiar to this track.

**Training reward is 10× the task reward, and the eval series changed meaning at commit
`971ab99` (2026-09-09).** `reward_scale = 10.0` (a brax default, carried over deliberately
so the new runs stay comparable with the 90-run cohort) is applied by
`RewardScalingWrapper` around the training env. Before `971ab99` the *eval* env inherited
the training wrappers, so `eval/*` was scaled too — and worse, `EpisodeWrapper.reset`
seeds `step_counter` to a random value in `[0, max_len/2)`, a phase spread that is correct
for training and wrong for measurement, so an eval episode could begin 499 steps in.
Measured on one WalkerWalk run at the moment of the fix:

| | before (wrapped eval) | after (bare eval) |
|---|---|---|
| lifespan | 755.1 ± 125.4 | 1000.0 ± 0.0 |
| episode reward | 1343.9 ± 221.3 | 157.1 ± 1.9 |

Note the reward *std* collapsing 100-fold: most of what looked like eval variance was
episode-length noise. **Never pool or contrast `eval/*` across `971ab99`.** Gate on the
**commit**, not `created_at`: runs from both sides were created on 2026-09-09, so a date
filter silently mixes them. `explicit-forward-model/extract.py` is the worked example —
it excludes twelve `d168093` cartpole runs, one of which reports 6147 where its post-fix
twins report ~830. This is the shape of the rodent's 2026-08-20 train-split bug, and it
hid for the same reason: "reward" was recorded where "which reward" was meant.
Training-side metrics are unaffected; the training stack was always
`RewardScalingWrapper → EpisodeWrapper → task`.

**There are two seeds, and `seed` is not the one PPO uses.** `train.py` draws the
network initialisation from the top-level `seed` (wandb `seed`, Hydra default 42) and
hands `train.seed` (wandb `config.seed`, Hydra default **1234**) to `train_ppo`, where it
keys the env resets, the rollouts and the eval episodes. A launch that passes only
`seed=N` therefore changes the init and leaves the PPO stream at 1234, and two batches
launched that way are *not* independent replicates of each other's rollouts. The
ReacherHard cohort has one batch of each kind: nine baseline runs at (init 42, ppo 1234)
because `seed=` was never passed, and ten forward-model runs at (init 42, ppo 42). Group
seeds on the **pair** — `f"i{seed}/p{config.seed}"`, as
[`reacher-hard-first-look/extract.py`](reacher-hard-first-look/extract.py) does — because
grouping on `seed` alone pools those two batches into one "seed 42" cell and reports a
cross-architecture contrast as if it were paired. Sweep scripts should pass both
(`seed=N train.seed=N`), which is what the later batches do.

**The control step is per task, not per track.** One delay step is 10 ms on one panel and
25 ms on the next:

| ctrl_dt | ms / step | tasks |
|---|---|---|
| 0.01 | 10 | CartpoleBalance, CartpoleSwingup, CheetahRun |
| 0.02 | 20 | BallInCup, ReacherEasy, ReacherHard, HopperStand, HopperHop, FingerTurnEasy, FingerTurnHard |
| 0.025 | 25 | WalkerStand, WalkerWalk, WalkerRun, HumanoidStand, HumanoidWalk |

`style.CTRL_DT_MS` is the rodent's 10 and `add_ms_axis(ax, max_x)` silently uses it, so
pass the task's own value — `add_ms_axis(ax, max_x, ctrl_dt_ms=25)`. Take it from
`env_params.ctrl_dt` in the run record rather than assuming a track-wide number; the
table above is the current registry, not a promise.

**The `history` and `timing` producers' WandB project is a spec field defaulting to the
rodent project.** `artifacts ensure --kind history` on a control-suite run list will fail
with `Could not find run …/nnx-ppo-rodent-delays/<id>`. Override it, and pin the resulting
spec id in the folder:

```bash
python -m vnl_experiments.artifacts ensure --kind history \
    --runs analysis/dm_control_suite/<q>/runs.csv \
    --set project='"emiwar-team/nnx-ppo-delays"'      # -> hist2000-8b97281e
python -m vnl_experiments.artifacts ensure --kind timing \
    --runs analysis/dm_control_suite/<q>/runs.csv \
    --set project='"emiwar-team/nnx-ppo-delays"'      # -> timing20000-3089bf7d
```

Because the project is hashed into the spec, this is safe rather than merely necessary:
control-suite and rodent curves get different spec ids and can never be pooled.

**The eval and video cadences changed on 2026-09-15, so curve *resolution* is a third
era boundary.** Runs before that date sampled `eval/*` every **600 k** steps and rendered
a video every **6 M**; runs after sample every **4.8 M** and **30 M**. Nothing else moved
— same `eval.n_envs` (256), same `eval.max_episode_length` (1 000), same reward, same
units, same optimisation — so the two eras are directly poolable for *values*, and only a
statement about the *shape* of a curve at fine step resolution needs gating on
`config.eval.every_steps` (read it off the run, not the date).

Why it changed: `eval.every_steps: 600_000` was brax's 60 M budget over the old script's
`num_evals = 100`, and it was never rescaled when `total_steps` was multiplied by 8 — so a
480 M run ran **800** evals where the number was chosen to give 100. That is not free. An
eval is 1 000 *sequential* steps over only 256 envs, which is latency-bound rather than
throughput-bound, so one eval costs more than one PPO step over 8 192 envs on every
locomotion task: on WalkerWalk / HumanoidWalk / HopperHop, 4–11 s against 0.6–2 s.
Measured across 116 runs (2026-09), **eval was 58–62 % of the wall clock and the PPO step
22–27 %**; video was 8–16 % (28–40 % on CartpoleSwingup, where the task is so cheap the
videos dominated); checkpointing was 0.1–0.9 % and was left alone. The new values return
61–66 % of a locomotion run's wall clock and give 100 eval points and 16 videos per 480 M
run. See [`where-the-wall-clock-goes/`](where-the-wall-clock-goes/) for the measurement
and `vnl_experiments/config/dmc_equivalence_test.py` for the pin.

If an eval curve looks noisy, raise `train.eval.n_envs` rather than lowering
`every_steps` again: more episodes inside one rollout is nearly free here, more rollouts
is not. And a run that "took much longer than expected" and predates 2026-09-15 is very
likely just the old cadence.

**CheetahRun's PPO step is ~10x slower than any other task's, and nobody knows why.**
9.6 s per iteration on an H200 against 0.77 s for WalkerWalk at the same `n_envs`,
`rollout_length` and network — while WalkerWalk does ten physics substeps per control step
to CheetahRun's one, and while CheetahRun's *eval* (256 envs) is ~5x faster than
WalkerWalk's. It is steady to ±1 % from the third iteration, so it is not compilation. A
CheetahRun run is therefore ~91 % PPO step, and none of the cadence advice above helps it.
`env_params.nconmax = 100000` with `njmax = 100` is an odd pairing beside WalkerWalk's
50000/100 and HumanoidWalk's 200000/250, and is the first thing to check. Budget ~10 h for
a 480 M CheetahRun until this is understood.

**Raw reward is not comparable across tasks.** Nine tasks with different reward
structures; a WalkerWalk return and a CartpoleBalance return are different quantities.
The shared preference for raw reward (§7) still holds — resolve it with **small multiples,
one panel per task**, not by dividing through by a per-task ceiling. If an across-task
summary really is needed, normalise there and only there, and say so in the caption.

**There is no clip split, so there is no held-out set.** No `TrainEvalSplit` tag, no
`old_eval` / `new_eval`: an eval episode is a fresh episode of the same task, differing
only in its seed. That makes "eval" a weaker word here than in the rodent track — it
measures the same distribution the policy trained on. Do not write "held-out" or
"generalisation" of a dm_control eval number.

**The eval env is deliberately unwrapped, so `episode_length` comes from the config.**
`EnvSpec.build(..., for_eval=True)` returns the bare registry env. dm_control registry
envs never terminate on their own, so an eval harness must impose its own horizon
(`train.eval.max_episode_length`, 1000); the training path gets it from `EpisodeWrapper`.
An eval that forgets the horizon runs forever rather than failing.

**Nine crashed runs in the legacy cohort were never diagnosed** (5 WalkerWalk, 4
Humanoid). If a selection turns up a hole there, that is why; the humanoids may simply
have exceeded their memory allocation. Treat them as attrition of unknown cause and check
whether it correlates with the swept variable before reading anything into a gap.
