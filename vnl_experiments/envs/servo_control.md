# Joint servo control (`servo_kp`) for dm_control_suite tasks

The `dm_control_suite` tasks ship with plain torque actuators: `<motor>` with a `gear`,
`gaintype=FIXED`, `biastype=NONE`, so joint torque is `gear * ctrl` and the policy's action
*is* the normalised torque. Setting `servo_kp > 0` converts them in place into position
servos, making the actuator an equilibrium-point controller:

```
tau = kp*(q* - q) - kv*qdot          q* = center + half*ctrl
```

The action keeps the same dimension and the same `[-1, 1]` range; only its *meaning* changes,
from "fraction of maximum torque" to "position within the joint's range".

`servo_kp = 0.0` leaves the model completely untouched — no field is written and
`mjx.put_model` is not re-run — so the torque baseline is the shipped env bit-for-bit, not a
servo configured to be neutral.

**Why the model and not a wrapper.** A Python servo in an env wrapper would run once per
`ctrl_dt` as a zero-order hold. The EP hypothesis is about *mechanics* — an instantaneous
spring — so the servo must run at `sim_dt`, inside every substep. Writing it into the
actuator is the only way to get that, and is what Playground itself does for Go1's `Kp`/`Kd`.
It is also much more stable numerically.

The four config fields, all recorded in `env_params`:

| field | default | meaning |
|---|---|---|
| `servo_kp` | `0.0` | dimensionless normalised stiffness; `0.0` = model untouched |
| `servo_damping_ratio` | `1.0` | ζ; `kv` is derived per joint, not swept |
| `servo_center` | `"qpos0"` | setpoint at `ctrl=0`; or `"range"` for the joint-range midpoint |
| `servo_unlimited_half_range` | `0.0` | `0.0` = unset; **raises** if a servoed joint is unlimited |

---

## 1. Why `servo_kp` is dimensionless

`servo_kp` is **not** a stiffness in N·m/rad. It is normalised per joint:

```
half[j]   = max(hi - center, center - lo)   # setpoint half-range, joint units (rad or m)
kp_ref[j] = gear[j] / half[j]               # stiffness saturating the torque limit
                                            #   at full-range position error
kp[j]     = servo_kp * kp_ref[j]
```

so that

> **`servo_kp = s` means the actuator saturates at a position error of `half/s`.**
> `s = 1`: full-range error saturates. `s = 10`: a tenth of the range saturates.

The reason is that gears differ by three orders of magnitude across these tasks, and by 4x
*within* Hopper:

| task | gears | joint half-ranges (rad) | `kp_ref` (N·m/rad) |
|---|---|---|---|
| ReacherEasy / ReacherHard | `0.05, 0.05` | π (unlimited, see §3.2), 2.793 | 0.016, 0.018 |
| HopperStand / HopperHop | `30, 40, 30, 10` | 0.524, 2.967, 2.531, 0.785 | 57, 13, 12, 13 |
| FingerTurnEasy / Hard | `30, 15` | 3.491, 1.920 | 8.6, 7.8 |

A single raw `kp` would therefore mean something ~1000x different on Reacher than on Hopper,
and ~4x different between Hopper's own hip and knee. A sweep grid chosen for one task would be
entirely off-scale on another, and the within-Hopper spread would mean "the stiffness sweep"
was really four different sweeps superimposed.

Normalising also removes a numerical-stability trap. Hopper and Reacher integrate with Euler
and `eulerdamp` disabled, so injected stiffness is integrated explicitly, with a stability
bound `kp < 4I/dt²`. At `sim_dt = 0.005` that is `servo_kp ≲ 600` (Hopper) and `≲ 1700`
(Reacher) — far beyond any plausible sweep. With a raw `kp`, a value of `1e4` looks
unremarkable on Hopper and diverges instantly on Reacher.

Convenient side-effect: substituting `kp = servo_kp * gear / half` into MuJoCo's actuator
algebra makes the gain **exactly** the parameter, `actuator_gainprm[0] = servo_kp`.

### 1.1 What the model fields become

MuJoCo's joint transmission has `length = gear*q` and `velocity = gear*qdot`, and
`qfrc_actuator = moment.T @ force` with `moment = gear`, so

```
tau = gear * (gainprm[0]*ctrl + biasprm[0] + biasprm[1]*gear*q + biasprm[2]*gear*qdot)
```

— note the affine bias picks up a **gear²**. Matching to
`tau = kp*(center + half*ctrl - q) - kv*qdot`:

```
gainprm[0] = servo_kp
biasprm[0] = servo_kp * center / half
biasprm[1] = -servo_kp / (gear * half)
biasprm[2] = -kv / gear**2
```

### 1.2 Force parity with the torque baseline

`forcerange = [-1, +1]`, `forcelimited = True`.

This is set so the servo has **exactly the torque authority the baseline has**, and not more.
The baseline motors have `gainprm[0] = 1` and `ctrl ∈ [-1, 1]`, hence `force ∈ [-1, 1]` and
`tau ∈ [-gear, gear]`. An unclamped servo's torque is unbounded, so without this a stiff arm
could beat the torque arm simply by being stronger — an artefact that would look exactly like
the hypothesis being confirmed.

The bound is on **actuator-space force, before the moment multiply** (MJX clips `force`, then
computes `qfrc_actuator = moment.T @ force`). A `[-gear, +gear]` bound would therefore cap
*joint torque* at `gear²`: 1600 N·m on Hopper's hip instead of 40, and 20x too weak on
Reacher. Because the error flips direction with gear, it would not look like a bug — it would
look like a finding.

Caveat for the writeup: the clamp buys equal **peak torque**, not equal **authority**. The
servo also changes the plant (it adds `-kp*q - kv*qdot` inside every substep). That is the
point of the experiment, not a confound, but do not describe the two arms as "the same
actuator with a different interface".

### 1.3 `servo_kp = 0` is not the limit of the family

`servo_kp = 0` is *torque control at full authority*. `servo_kp = 0⁺` is an essentially **free
joint** — the actuator force goes to zero. The parameter is a mode switch wearing a continuous
parameter's clothes, and the servo family does not connect continuously to the baseline at
zero.

**For plots:** use a log-spaced `servo_kp` grid and draw the torque baseline as a horizontal
reference line. A reward-vs-`servo_kp` plot with the baseline at `x = 0` on a linear axis
reads as a continuous curve through a point that is not on it.

At the other end, `servo_kp → ∞` with the force clamp is bang-bang position control, not a
return to torque control.

---

## 2. Damping is derived, not swept

```
kv[j] = 2 * servo_damping_ratio * sqrt(kp[j] * I_eff[j])
I_eff[j] = mj_model.dof_M0[jnt_dofadr[j]]
```

`dof_M0` is the diagonal of the mass matrix at `qpos0` and **already includes armature** —
verified on Hopper, where `dof_M0[6] = 0.2155` against `dof_armature[6] = 0.2`. Armature
dominates the effective inertia of Hopper's distal joints, so using a bare link inertia here
would be materially wrong.

**Why derived.** With a fixed `kv`, the damping ratio

```
zeta = kv / (2*sqrt(kp*I))
```

falls as `1/sqrt(kp)`. The stiff end of a sweep would then be systematically underdamped and
oscillatory, and would degrade for a reason with nothing to do with sensorimotor delay. The
`servo_kp` axis would be measuring "stiffness *and* progressive loss of damping" at once —
a textbook confound for precisely this experiment. Deriving `kv` holds ζ fixed so that
`servo_kp` is the only thing varying.

Default `servo_damping_ratio = 1.0`, critically damped. That is also the right *physical*
default for the EP framing: an equilibrium-point controller is normally posed as a
spring-damper whose equilibrium the controller shifts, and ζ=1 is the non-ringing reference.

### 2.1 The interpretable axis

With ζ=1 the servo has a natural frequency and settling time

```
omega_n[j] = sqrt(kp[j] / I_eff[j])
t_settle   ~ 4 / omega_n[j]        seconds
```

which is directly commensurate with the network delay, `k * ctrl_dt` seconds. `servo_report()`
logs both per joint into the run record.

**This is the plot to make.** "Servo settling time vs sensorimotor delay, both in ms" is an
interpretable 2-D surface; "arbitrary stiffness units vs delay steps" is not. The hypothesis
in those terms is that the servo helps when `t_settle < k*ctrl_dt` — when the periphery
corrects faster than the central loop can.

---

## 3. Env-specific limitations and caveats

### 3.1 Reward terms that read the action

Some tasks compute part of the reward from `action`. Under the servo `action` is a
**setpoint**, not a torque, so such a term silently changes meaning — typically from
"penalise effort" to "stay near the neutral posture". **The objective is then not the same at
`servo_kp = 0` and `servo_kp > 0`, and reward is not comparable along the axis.**

| task | action-dependent? | note |
|---|---|---|
| `HopperHop` | no (`del action`) | clean |
| `ReacherEasy`, `ReacherHard` | no | clean |
| `FingerTurnEasy`, `FingerTurnHard`, `FingerSpin` | no | clean |
| `WalkerStand/Walk/Run`, `CheetahRun`, `BallInCup` | no | clean |
| **`HopperStand`** | **yes** | `_stand_reward` multiplies by `small_control = tolerance(action).mean()`, rescaled to `[0.8, 1.0]` |
| **`HumanoidStand`, `HumanoidWalk`** | **yes** | `small_control` factor |
| **`CartpoleBalance`, `CartpoleSwingup`** | **yes** | `small_control` and `action_penalty` |

Prefer the clean tasks for any headline stiffness sweep. For the affected ones,
`reward/small_control` is logged as its own metric, so the action-independent factor can be
reported separately — do that rather than comparing totals.

### 3.2 Reacher: the shoulder is unlimited

`ReacherEasy` / `ReacherHard` joint 0 has `jnt_limited = False` — unlimited rotation, so there
is **no natural setpoint range**. `apply_servo` raises unless `servo_unlimited_half_range` is
set explicitly; the reacher group files set it to π.

π is not invented: `reacher.py` resets the shoulder uniform on `[-π, π]`, so that is the env's
own notion of the joint's range. It is still a modelling decision, which is why it is an
explicit config value recorded in `env_params` rather than a silent default buried in a
helper.

Two consequences that belong in any report using Reacher:

1. MuJoCo's affine bias is linear in `length`, so this is an **absolute-angle torsion spring**,
   not a shortest-angle position servo. There is no wrapped-error formulation available
   through `gainprm`/`biasprm`. Winding past π keeps increasing the restoring torque until the
   force clamp.
2. Therefore at `servo_kp > 0` the shoulder is **effectively range-limited by the servo**: the
   servo removes a degree of freedom the torque baseline has. **Reacher is the one env where
   `servo_kp` changes the configuration space, not just the actuator.** If the headline claim
   is "stiffness compensates for delay", this is where a reviewer will push.

### 3.3 Hopper: `qpos0` lies outside the knee's joint limit

Hopper's knee has `qpos0 = 0.0` but `jnt_range = [0.087, 2.618]` — the model's reference value
violates the joint's own limit. `center` is therefore **clamped**,
`center = clip(qpos0, lo, hi)`, giving 0.087 (extended leg, a sensible neutral for a hopper).
Without the clamp the servo would push against the limit constraint permanently at `ctrl = 0`.

The clamp leaves the knee's setpoint map asymmetric: `center = 0.087`, `half = 2.531`, so
`ctrl < 0` maps below the lower limit and is **dead**. See §3.4a for how much this costs.

Verified by rendering at `ctrl = 0`: under `qpos0` the hopper holds a straight leg (knee 5°),
under `range` a deep crouch (hip −68°, knee +85°) that fights the `standing` reward term.
Both settings are recorded, so the choice is testable rather than assumed.

### 3.4 Finger: `ref="-90"` puts `qpos0` near a limit

`finger.xml` sets `ref="-90"` on the `proximal` joint, so `qpos0 = -1.571` within
`jnt_range = [-1.920, 1.920]` — near the lower limit. Hence `half = 3.491` and, as with
Hopper's knee, much of the negative `ctrl` range maps outside the limit and is dead.

The algebra is unaffected (`length = qpos*gear` has no `ref` subtraction), but the neutral
posture is 90° from the pose the model is drawn in: rendered at `ctrl = 0`, the digit points
straight up, away from the spinner it is supposed to turn.

### 3.4a How much action range the `qpos0` centre costs

`servo_center = "qpos0"` is a posture prior, but it is also a **handicap on the servo arm**:
any `ctrl` whose setpoint falls outside the joint's own limits is dead, because the joint
cannot go there. Measured at `servo_kp = 1`:

| task | joint | usable `ctrl` band under `qpos0` | dead | under `range` |
|---|---|---|---|---|
| ReacherEasy / ReacherHard | shoulder, wrist | `[-1.00, +1.00]` | 0% | 0% |
| HopperStand / HopperHop | waist, ankle | `[-1.00, +1.00]` | 0% | 0% |
| HopperStand / HopperHop | **hip** | `[-1.00, +0.06]` | **47%** | 0% |
| HopperStand / HopperHop | **knee** | `[+0.00, +1.00]` | **50%** | 0% |
| FingerTurnEasy / Hard | **proximal** | `[-0.10, +1.00]` | **45%** | 0% |
| FingerTurnEasy / Hard | distal | `[-1.00, +1.00]` | 0% | 0% |

Reacher is unaffected — `qpos0` is already its range midpoint on both joints — so the two
centre modes are identical there. Hopper and Finger lose roughly half the policy's output
range on their most task-relevant joints.

`servo_center = "range"` makes every band exactly `[-1, 1]` and costs the posture prior
instead. **Neither is free**, and which one is the confound depends on the claim: `qpos0`
gives the servo arm a smaller effective action space than the torque arm, while `range` gives
it a different and possibly adversarial neutral posture.

`servo_report()` records `usable_ctrl_lo` / `usable_ctrl_hi` / `dead_ctrl_fraction` per joint
for exactly this reason — a reward comparison would otherwise absorb the handicap silently.
Consider running the pilot at both settings on HopperHop before committing to one.

### 3.5 The neutral action is a posture prior

Under torque control `action = 0` means "no torque". Under the servo it means "hold the
reference posture at full stiffness". A freshly initialised PPO policy has mean ≈ 0, so the two
arms start from completely different behaviour, which changes exploration and early return.

This is arguably part of the effect being measured rather than a confound — but it is a
*choice*, it interacts with §3.3/§3.4, and it should be stated rather than discovered.

### 3.6 Divergence bookkeeping at high stiffness

`NaNGuardWrapper` is on the **train env only**; the eval env is deliberately bare. A stiff
servo raises the divergence rate, and the two sides fail differently:

- train: a divergence becomes a silent zero-reward termination, so a high-`servo_kp` arm can
  read as "bad at the task" when it is really "diverging";
- eval: no guard at all, so a divergence produces a raw NaN in the published numbers.

**Check `diagnostics/nonfinite_next_obs` per `servo_kp` as part of the protocol.**
`registry.py` already notes that the no-guard-on-eval decision should be revisited if the
divergence rate rises; a stiffness sweep is exactly that circumstance.

### 3.7 Training budget

`train=dmc` pins `total_steps: 480M` for every task. `HopperHop` is materially harder than
`WalkerWalk`, so its `servo_kp = 0` point may be undertrained — which would masquerade as a
stiffness effect. Keep the budget for cohort comparability, but check that the learning curves
have plateaued before reading anything off the `servo_kp` axis.

---

## 4. What is recorded

`env_params` in both `config.json` and the WandB config carries the whole env ConfigDict, so
all four `servo_*` fields are recorded automatically and are selectable as
`env_params.servo_kp` etc. in the analysis extracts.

`servo_kp` alone does **not** determine the physics, though — the per-joint `kp`, `kv`,
`center`, `half` depend on the model. `servo_report(env)` puts those, plus `omega_n` and
settling time per joint, into `wandb_config` so the physics is recoverable from the run record
without re-deriving it from the XML.

Runs with `servo_kp > 0` carry a `kp{value}` tag and a `_kp{value}` suffix in the run name;
runs at `servo_kp = 0` are named exactly as before, so the existing `nnx-ppo-delays` cohort
stays name-compatible.
