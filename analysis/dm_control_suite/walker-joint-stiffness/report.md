# WalkerWalk: is a joint position servo a viable action space, and at what stiffness?

## Question

`servo_kp` turns WalkerWalk's torque motors into equilibrium-point position servos
(`vnl_experiments/envs/servo_control.md`). Before any test of the hypothesis that
peripheral stiffness absorbs sensorimotor delay, this pilot asks the prerequisite: **is the
servo trainable at all, over what stiffness range, and at what cost relative to torque
control at matched delay?**

An answer is a stiffness band within which the servo reaches torque-level reward, plus an
honest statement of what the delay axis can and cannot support.

**Headline: it is viable above `servo_kp ≈ 4`, it costs nothing at delay 0, and it learns
several times more slowly. The delay question is not answered, and could not have been —
see "What this pilot cannot answer".**

## Dataset & comparability

- **Source:** WandB `emiwar-team/nnx-ppo-delays`, one launch (all 18 runs carry the note
  *"Sweeping kp."*), selected by the `CONDITIONS` in `extract.py` and frozen in `runs.csv`.
  Selection is on the *configuration* — task, network, budget, `servo_center` — never on
  the run name or the note.
- **Conditions:**

| condition | task | n | seeds | `servo_kp` | delays | network | commit |
|---|---|---|---|---|---|---|---|
| `torque` | WalkerWalk | 6 | 50, 51, 52 | 0 | 0, 5 | DelayedMLP | `7bedb5cc` |
| `servo` | WalkerWalk | 12 | 50 (52 at kp=1) | 0.25–16 | 0, 5 | DelayedMLP | `7bedb5cc` |

- **Reward reported:** `eval/episode_reward/mean`, i.e. **unscaled and full-length** —
  every run is post-`971ab99`, so the eval env is bare and none of the 2026-09-09
  scaled/truncated era is present. Recorded as `reward_source = dmc_eval` in `data.csv`.
  Reduced to `reward_final`, the **mean of the eval points in the last 5e7 of 4.8e8 steps**
  (~11 points at the 4.8e6 cadence), not the run's final point (README §6). All y-axes are
  raw reward; no ratio or normalised axis is used anywhere.
  Note this track has no held-out set: a dm_control "eval" is a fresh episode of the same
  task, so none of these numbers is a generalisation measure.
- **Tasks:** WalkerWalk only. No cross-task comparison is made.
- **Artifacts used:** `REQUIRES = ["index", "history:hist2000-8b97281e"]`. **Coverage is
  18/18 in both conditions** (`coverage.txt`); no run is dropped for a missing artifact,
  and every `history` artifact reaches `max_step = 480215040`, i.e. none is a mid-training
  snapshot.
- **Programmatic comparability:** `comparability.txt` flags **nothing** — all 29 invariants
  are single-valued both within each condition and pooled, including every PPO
  hyperparameter, `reward_scale`, `episode_length`, `ctrl_dt`/`sim_dt`, the network sizes,
  the eval cadence and shape, and `servo_damping_ratio` (1.0, critically damped) and
  `servo_center` (`range`). `summary._step` is `480215040` for all 18, so no run was
  preempted short.
- **Manual comparability:**
  - Configs inspected directly from the index, not via tags — the initial listing was
    filtered on `env_params.servo_kp` alone, which pulled in HopperHop and HumanoidWalk
    runs at unrelated budgets whose low reward looked like a WalkerWalk regression. `env`
    is the discriminating column.
  - Tags and notes read on all 18. The `kp{value}` variant tags are present and correct.
    Every servo run also carries `env-override`, which is expected here (the launch sets
    `env.servo_kp=`) and is **not** a warning in this cohort.
  - `git_commit` is `7bedb5cc` for all 18, so no cross-commit diff is required — **but see
    the dirty-tree caveat below, which is the reason that is not the comparability
    argument.**
  - Environment is uniform: one OS build, CUDA 13.3 across all 18.

- **Caveats:**
  1. **`repos.vnl_experiments.dirty == True` on all 18 runs, so `git_commit` does not
     identify the code that ran** (README §4) — and the servo was under active development
     on the launch days. Addressed behaviourally in `servo_identity.py` / `.txt`, which
     checks what the runs actually recorded: the derived per-joint `kp_ref = kp/servo_kp`
     is **identical to 6 decimal places across all 12 servo runs** (95.492966 / 38.197186 /
     25.464791 N·m/rad for hip / knee / ankle) and matches the documented `gear/half`;
     `dead_ctrl_fraction` is 0 on every joint; and torque runs carry no servo parameters at
     all, confirming `servo_kp = 0` left the model genuinely unpatched. **The actuator was
     identical in every run of this cohort.** This does not prove the *training* code was
     unchanged — nothing can, with a dirty tree.
  2. **Ten of the thirteen cells are n=1.** Only `servo_kp = 0` has three seeds; `kp = 1`
     has two at delay 0. Variance is strongly regime-dependent: in the solved regime the
     seed spread is ≤ 3.3 (torque), but at `kp = 1`, delay 0 the two seeds are **476 and
     819 — a spread of 343**. No single-cell difference in the transition region is
     resolved by this data.
  3. **Design hole:** `servo_kp = 0.25` was run at delay 5 only.
  4. **One run failed** (`ujoqf3rj`, `kp = 1`, delay 0, seed 51) at `_step = 0` — it never
     started. Attrition unrelated to any condition, but it is why `kp = 1`, delay 0 has two
     seeds rather than three.
  5. **Five GPU models** across 18 runs (Blackwell 7, H200 5, A100-80 3, A100-40 2, A40 1),
     scattered across conditions rather than confounded with `servo_kp`. Per README this is
     a throughput confound, not a reward one, and no timing claim is made here.
  6. **No divergence diagnostic was logged**, so the cause of the `kp = 8` collapse
     (finding 4) is unidentified.

## Figures

![End-of-training reward against servo_kp, one panel per delay](figures/stiffness.png)

The shape of the stiffness axis. Torque is the dashed reference line, not a point at x=0 —
the servo family does not connect continuously to it, since `servo_kp = 0` is torque
control at full authority while `servo_kp → 0⁺` is an essentially free joint. **Look at the
rings:** they mark runs that were still climbing (|late gain| > 20) or had collapsed
(drawdown > 50), i.e. runs whose end-of-training reward is not a converged level. At delay
5 that is every servo point except the one at the bottom.

![Eval-reward learning curves](figures/curves.png)

Why the ringed points are ringed. The torque arm (orange) plateaus by ~30e6 steps in both
panels. Every servo curve is still rising at 4.8e8, and at delay 5 the stiff ones
(`kp = 8`, `16`) show sharp collapses — the deep notch near 460e6 is `kp = 8` falling ~360
points immediately before the readout window.

![Late gain and max drawdown against servo_kp](figures/convergence.png)

The same two disqualifications as numbers per run. Left: reward gained over the last 1e8
steps, where torque sits at ~0 (converged) and `kp = 4` at delay 5 sits at **+133**. Right:
largest peak-to-trough fall in the last 2e8 steps on a log axis — torque is 1.9–4.3
throughout, `kp = 8` at delay 5 is **361**.

## Tentative conclusion

**What the data shows:**

1. **The servo is viable above `servo_kp ≈ 4`, and costs nothing at delay 0.** At delay 0,
   `kp = 16` reaches 976.7 against torque's 976.6 — indistinguishable — and `kp = 4` and `8`
   reach 958 and 965. Equilibrium-point control is therefore not intrinsically a harder
   action space on this task, once stiff enough.
2. **Below `kp ≈ 4` it fails, and the failure is graded.** 648 at `kp = 1` (seeds 476/819),
   145 at `kp = 0.5`, 60 at `kp = 0.25`. The two softest are near-flat in their last 1e8
   steps (+6.6 and +0.4), so those look like genuine floors rather than slow learning; the
   `kp = 1` runs were still climbing (+36, +52) and are not resolved.
3. **The servo arm learns several times more slowly than torque.** Torque plateaus by
   ~30e6 steps; the servo needs 250–350e6 at delay 0 and has *not* converged at 4.8e8 at
   delay 5. This is README §6's common-budget trap, and here it points **against** the
   servo: a shared 4.8e8 readout understates it.

**What it suggests but does not establish:** that `kp = 4` may be the sweet spot — it was
the only stiffness both stable (drawdown 1.0) and still improving steeply (+133) at delay
5, ending at 955 against torque's 964. With more budget it looks likely to match torque.
This is n=1 and should not be quoted as a result.

**What this pilot cannot answer — and the most useful thing it produced:**

The delay question is untestable from these runs, for two independent reasons.

- **There was almost no delay damage to repair.** Torque control loses 1.3 % from delay 0
  to delay 5 (976.6 → 964.0). A manipulation that is supposed to *recover* delay-induced
  loss cannot be measured against a 1.3 % loss.
- **The pilot never entered the regime the hypothesis is about.** The prediction is that
  the servo helps once it is *faster than the delayed central loop*. WalkerWalk's slowest
  joint (hip) settles in 195.7 ms at `kp = 16` — still **slower** than the 125 ms delay
  being tested. Not one run in this cohort had a servo faster than its own delay
  (`servo_identity.txt` §4).

Since settling time scales as `kp^(-1/2)`, the crossover is:

| delay | | `servo_kp` for the hip servo to beat the loop |
|---|---|---|
| 5 | 125 ms | **> 39** |
| 10 | 250 ms | > 10 |
| 15 | 375 ms | > 4.4 |
| 20 | 500 ms | > 2.4 |

So the stiffness range chosen here (0.25–16) is well matched to **delays 15–20** and far
too soft for delay 5. Read that way the pilot is a success: it says where the next sweep
has to sit.

## Follow-ups

- **Move the sweep to delay 15–20, keeping `servo_kp` 4–16.** That is the one region where
  torque control is meaningfully hurt (delay 20 torque: 577 and 941 at 4e9 steps, outside
  this cohort) *and* the servo is faster than the central loop. This is the cheapest path
  to a real test.
- **Raise the budget past 4.8e8 for any servo run.** The servo arm had not converged in any
  delay-5 cell. Until it does, no stiffness ordering is readable, and a shared budget
  penalises the servo specifically. The sibling `walker-forward-model-1b/` faced exactly
  this and resolved it by moving to 1e9.
- **If delay 5 is kept, add `servo_kp` 32 and 64** — otherwise the hypothesis has no room
  to be right there.
- **Three seeds minimum in the transition region (`kp` 1–8).** The 343-point spread at
  `kp = 1` is the noise floor that any claim about the knee of this curve has to clear.
- **Log the divergence diagnostic** (`diagnostics/nonfinite_next_obs`) and re-examine the
  `kp = 8` collapse. A 361-point drawdown in a stiff servo is what the servo_control.md
  §3.6 warning predicts, and it is currently unattributed.
- **Worth testing once:** whether the servo's slow learning is an exploration-scale
  artefact. Under the servo the action is a position setpoint, so the policy's initial
  `min_std = 0.001` means something quite different from what it means for a torque.

---

*Reproduce:* `../.venv/bin/python analysis/dm_control_suite/walker-joint-stiffness/extract.py && ../.venv/bin/python analysis/dm_control_suite/walker-joint-stiffness/plot.py`
(add `--sync --refresh` to the extract to pull in runs added since `runs.csv` was frozen;
`../.venv/bin/python analysis/dm_control_suite/walker-joint-stiffness/servo_identity.py`
re-runs the actuator-identity check.)
