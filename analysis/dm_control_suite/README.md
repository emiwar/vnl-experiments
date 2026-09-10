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
| Control step | `ctrl_dt = 0.025 s` — 1 step = **25 ms**, *not* the rodent's 10 |
| Tasks | `WalkerWalk` `WalkerRun` `WalkerStand` `CheetahRun` `HumanoidWalk` `HumanoidStand` `CartpoleBalance` `CartpoleSwingup` `BallInCup` |
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

## There are two eras of run, and they do not share a schema

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

**Training reward is 10× the task reward, and the eval series changed meaning on
2026-09-10.** `reward_scale = 10.0` (a brax default, carried over deliberately so the new
runs stay comparable with the 90-run cohort) is applied by `RewardScalingWrapper` around
the training env. Until 2026-09-10 the *eval* env inherited the training wrappers, so
`eval/*` was scaled too — and worse, `EpisodeWrapper.reset` seeds `step_counter` to a
random value in `[0, max_len/2)`, a phase spread that is correct for training and wrong
for measurement, so an eval episode could begin 499 steps in. Measured on one WalkerWalk
run at the moment of the fix:

| | before (wrapped eval) | after (bare eval) |
|---|---|---|
| lifespan | 755.1 ± 125.4 | 1000.0 ± 0.0 |
| episode reward | 1343.9 ± 221.3 | 157.1 ± 1.9 |

Note the reward *std* collapsing 100-fold: most of what looked like eval variance was
episode-length noise. **Never pool or contrast `eval/*` across 2026-09-10.** Add
`created_at` to `INVARIANTS` for any cohort that straddles it, and say which side of it a
number came from — this is exactly the shape of the rodent's 2026-08-20 train-split bug,
and it hid for the same reason: "reward" was recorded where "which reward" was meant.
Training-side metrics are unaffected; the training stack was always
`RewardScalingWrapper → EpisodeWrapper → task`.

**1 control step is 25 ms here.** `style.CTRL_DT_MS` is the rodent's 10, so
`add_ms_axis(ax, max_x)` on a dm_control figure mislabels the top axis by 2.5× and raises
nothing. Always pass `add_ms_axis(ax, max_x, ctrl_dt_ms=25)`.

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
