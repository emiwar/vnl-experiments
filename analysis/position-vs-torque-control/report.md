# Position vs torque control — is imitation easier, and does the forward model still win?

## Question
The rodent's actuators can be driven in two modes (`env_params.torque_actuators`):
**torque control** (`True`, the project default — motors converted to torque-mode actuators)
or **position control** (`False` — MuJoCo position/PD actuators, where the network commands a
target joint configuration and a built-in PD loop supplies the torque). Position control is
presumably *easier* — the PD loop does part of the stabilisation. Two questions:

1. **Is position control easier / more robust to observation delay than torque control?**
2. **Is an explicit forward model still advantageous under position control?**

## Dataset & comparability
A 2×2 (control-mode × network) design over an efference-matched delay sweep
(`efference_length == delay_k`), seed 42, standard architecture (enc `[512]×4`, dec `[512]×4`,
critic `[1024,1024]`, `latent_size=32`, `kl_weight=0.001`), **frame held at `current_root`**,
all restored at `actual_step = 600,064,000`, train clip 250 frames.

| condition | control mode | network | delays | commit |
|---|---|---|---|---|
| `pos_efference` | position (`torque_actuators=False`) | efference EncDec | 0,2,5,10,20,50,70,100 | `891cd0d3` |
| `pos_forward_model` | position (`torque_actuators=False`) | explicit FM (`fm_loss_weight=1`, detached) | 0,2,5,10,20,50,70,100 | `891cd0d3` |
| `torque_efference` | torque (`torque_actuators=True`) | efference EncDec | 0,2,5,10,20,50 | `1cd5838f` (canonical eff sweep) |
| `torque_forward_model` | torque (`torque_actuators=True`) | explicit FM (`fm_loss_weight=1`) | 0,2,5,10,20,50 | `54643764` (canonical FM sweep) |

- **Authoritative control mode** is read from `env_params.torque_actuators`; network type from
  run **tags** (`ForwardModel`/`EncDec`). Selection and columns are in [extract.py](extract.py).
- **Frame is held constant** — every condition (position *and* torque) is `current_root`, read
  from `env_params.body_target_frame`, so the frame is *not* a confound here (it is a verified
  single-valued invariant in [comparability.txt](comparability.txt)).
- **Comparability: all invariants single-valued within every condition** (frame, latent/kl/
  enc/dec/critic sizes, seed, clip_length, and the trained step `actual_step` — see
  [comparability.txt](comparability.txt)). The only things that vary are the experimental axes
  (`control_mode`, `network`, `delay_k`) and, by design, `git_commit`.
- **The control mode is confounded with the code commit** (position runs at `891cd0d3`; torque
  baselines at `1cd5838f`/`54643764`). This was checked by hand and is **behaviourally inert**:
  - `891cd0d3` differs from `909e774d` (the reference-root cohort) **only by
    `eval_runs.txt`** — no training/env/network/reward code — so its training code is identical
    to that cohort's.
  - The reference-root analysis already certified `909e774d`'s efference and FM training as
    behaviourally identical to these baselines except for the added, default-inert options
    (`inject_key=None` in `efference_copy.py`; `detach_prediction=True` in `forward_model.py`,
    which reproduces the original hardcoded `stop_gradient`). Everything else in the diff is
    eval/analysis/video tooling, not used during training.
  - The position runs record HEAD `891cd0d3` but were launched from a working tree with the
    experimental edits applied — `env_params` confirms `torque_actuators=False` **and**
    `body_target_frame=current_root` (both differ from that commit's script defaults). The
    recorded value that matters — the actuator mode — is read authoritatively from `env_params`.

  So within each network the *only* training-relevant difference between the position and
  torque runs is the actuator mode itself.
- **Metric.** End-of-training **eval-on-train-clips** episode reward (the training script sets
  `eval_env = train_env`); the "new logging api" rename `episode_reward/mean` →
  `eval/episode_reward/mean` is coalesced in `extract.py`. `torque_actuators` changes the
  *actuators*, not the imitation reward shaping — the tracking terms (joints, root position/
  orientation, end-effector, torso-height) are control-mode-agnostic and dominate. The only
  mode-sensitive reward terms are the small penalties `energy_cost`, `control_cost` and
  `control_diff_cost`, which together are **~3 % of the per-step reward** (energy_cost ≈
  −0.012, control_cost ≈ −0.12, control_diff_cost ≈ −0.006, vs a total ≈ 4.4 /step). So the
  reward is a fair overall-performance measure across modes, with a small caveat that a few-%
  slice of any cross-mode difference is the differently-scaled cost terms rather than tracking.
- **Caveats.** Single seed per cell (differences are indicative, not seed-significant). Reward
  is eval-on-train-clips only — these runs have **no offline batch eval yet** (no held-out /
  long-clip `new_eval`, no termination-mode breakdown). The duplicate delay-0
  `torque_efference` run is collapsed by keeping the higher-reward row in `plot.py`. Delays 70
  and 100 exist only for position control (torque was not swept that far).

## Figures
Q1 — within each network, position control (solid) barely degrades across the whole delay
sweep, while torque control (dashed) collapses. At zero delay the two modes are essentially
tied (torque marginally higher).

![Position vs torque reward vs delay, faceted by network.](figures/control_comparison.png)

Q2 — under position control the explicit forward model (green) tracks the efference decoder
(orange) closely and edges ahead only at long delay (left). The forward-model *advantage*
(FM − efference, right) is **far smaller under position control (blue) than under torque
control (red)**: torque climbs to +300 by delay 50, position reaches only ~+35 at delay 50 and
~+100 by delay 100.

![FM vs efference under position control, and the FM advantage for both control modes.](figures/fm_advantage.png)

## Tentative conclusions

**Q1 — position control is dramatically more robust to observation delay; at zero delay it is
marginally *worse* than torque control.** Reward Δ(position − torque):
- *At delay 0:* −48 (efference), −39 (forward model) — torque is ~2 % higher when there is no
  delay (the extra authority of direct torque control buys a little peak tracking).
- *As delay grows this flips hard:* +85/+395/+761 (efference at delay 10/20/50) and
  +16/+277/+496 (forward model at delay 10/20/50). Position-control efference loses only ~150
  reward (1958 → 1806) from delay 0 → 50, whereas torque-control efference loses ~960
  (2006 → 1045).

The PD loop absorbs most of the delay penalty: with position targets, a stale observation
still commands a sensible set-point that the local controller drives toward, whereas a stale
*torque* is applied open-loop for the whole delay. So "position control is easier" is really
"position control is far more **delay-tolerant**", not uniformly higher reward.

**Q2 — yes, the explicit forward model is still advantageous under position control, but the
benefit is much smaller and appears later than under torque control.** FM − efference:
- *position:* −11 (0), −4 (5), −3 (10), **+33 (20), +36 (50), +56 (70), +97 (100)**.
- *torque:* −20 (0), −2 (5), **+66 (10), +151 (20), +301 (50)**.

Both curves share the same shape — a small penalty at short delay, a crossover, then a growing
gain — but position control shifts the crossover later (~delay 10–20 vs ~5–10) and flattens
the gain by roughly an order of magnitude. This is the expected consequence of Q1: the PD loop
already handles most of the delay, so there is much less residual delay for an internal
forward model to compensate, and the explicit predictor adds only a modest edge (which
nonetheless keeps growing out to the longest delays tested). The FM's own prediction error
rises with delay and plateaus (`fm_pred_mse` ≈ 0.0005 → 0.034 → 0.057 → 0.061 at delays
0/2/20/100).

**Bottom line.** Position control makes delayed imitation far easier — it is nearly
delay-invariant out to 100 steps (1 s) where torque control has long since collapsed — at the
cost of a slight peak-performance dip with no delay. The explicit forward model is still the
better choice at longer delays, but under position control its advantage is small (tens of
reward) rather than the hundreds it buys under torque control.

### Follow-ups
- **Offline batch eval** (`eval_runs.py`) on these 16 position runs to add `old_eval`/
  `new_eval` generalization, termination modes and hazard — the training-clip reward here can't
  speak to long-clip robustness. (Not yet in `eval_runs.txt`; the 16 ids are in
  [data.csv](data.csv), all `env_class=AbsoluteImitation`.)
- **Multiple seeds** per cell to turn the indicative long-delay position FM edge into a real
  signal or noise, and to confirm the small zero-delay torque advantage.
- Extend the **torque** sweep to delays 70/100 to complete the 2×2 grid (torque currently stops
  at 50, where it has already degraded far below position control).
