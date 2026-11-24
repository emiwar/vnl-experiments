# Weights strand: notes (input to report.md)

Source: `weights.py` → `weights_features.csv`, `weights_similarity.csv`, `weights_classify.csv`,
`figures/weights_*.png`. Cohort: the 62 complete runs of `../walker-terrain-factors/` at the
480 215 040 checkpoint. All quantities are about **ΔW = W − W0**, where W0 is each run's init
rebuilt as `train.py` builds it (`common.init_arrays`). Trained 256×256 layers still have a
cosine of ≈0.86–0.89 with their own init (`checks.txt`).

## Init sharing (verified)
- Actor/decoder layers 1–4 have **bit-identical init in all 393 same-seed pairs**, FM decoder
  vs MLP actor included. Their hidden units are therefore aligned within a seed.
- Critic layers 1–5 share init in only 192/393 same-seed pairs (it depends on architecture).
  `cos_critic` is NaN elsewhere.

## 1. How far each layer moved (‖ΔW‖/‖W0‖, mean)
| layer | FM torque | FM servo | MLP torque | MLP servo |
|---|---|---|---|---|
| predictor 0–4 | 0.99–1.40 | 0.88–1.21 | – | – |
| actor/decoder 0 | 0.22 | 0.21 | 0.28 | 0.24 |
| actor 1 / 2 / 3 | 0.64 / 0.53 / 0.45 | 0.61 / 0.50 / 0.40 | 0.56 / 0.48 / 0.44 | 0.53 / 0.45 / 0.38 |
| head (actor 4) | 0.46 | 0.53 | 0.53 | 0.57 |
| critic 1 → 5 | 1.2 → 9.5 | 1.2 → 13.2 | 1.3 → 12.3 | 1.2 → 15.7 |

- The forward-model predictor (trained by L2) moves more than the policy, by about its own
  init norm. The policy's first layer moves least (0.2–0.3).
- Servo moves the hidden actor layers less and the head more. Its head update is also more
  low-rank: actor4 ΔW top-singular-value fraction is 0.39–0.40 for servo against 0.30–0.33
  for torque.

## 2. Input reliance (first-layer row norm / init row norm)
| | eff lag1 | lag2 | lag3–5 | orient | vel | sensor |
|---|---|---|---|---|---|---|
| MLP actor, torque | 1.09 | 1.04 | 1.01 | 1.12 | 1.13 | 0.99 |
| MLP actor, servo | 1.09 | 1.05 | 1.02–1.03 | 1.09 | 1.08 | 1.06 |
| FM predictor, torque | **0.76** | 0.66 | 0.71–0.93 | 1.53 | 1.46 | 1.00 |
| FM predictor, servo | **1.29** | 1.05 | 0.80–0.81 | 1.35 | 1.22 | 0.99 |

- The **MLP actor barely re-weights the efference queue** (1.01–1.09 × init, monotone in
  lag). By this measure it does not single out recent actions much more than old ones.
- The **FM predictor reorganises its input strongly**. It grows the observation rows
  (1.2–1.5×) and *shrinks* most efference rows below init. With a **servo it up-weights
  the newest action** (1.29 vs 0.76 for torque) and down-weights velocity. This fits the
  physics: a servo action is a target joint angle that the body reaches within a few
  steps, so the last command predicts the current posture; under torque control the
  action is integrated twice and says less, per lag, about the state.
- The block-sensor rows stay at ≈1.0× init everywhere. The first-layer weights barely
  changed for the sensor, even though the sensor raises reward. Gradient attribution is
  the better way to measure how much the policy uses it.
- Head: the learned log-std at the bias is ≈ −0.4 for every condition (no difference). The
  action-mean head's **knee** columns grow less with a servo (MLP 1.04 vs 1.22, FM 0.99
  vs 1.10 × init).

## 3. Within-seed ΔW similarity (actor/decoder 1–4, aligned by shared init)
| pair type | n | median cos |
|---|---|---|
| replicate (identical config and seeds) | 8 | 0.59 (range 0.27–0.69) |
| differ only in privileged critic | 10 | 0.35 |
| only sensor | 21 | 0.33 |
| only efference | 15 | 0.26 |
| only servo | 24 | 0.17 |
| only forward model | 18 | 0.15 |
| 2+ factors | 296 | 0.14 |

- GPU nondeterminism alone takes identical runs to cos ≈ 0.6, and one replicate pair goes
  as low as 0.27. Every single-factor pair sits below the replicate median.
- In an OLS fit on the non-replicate pairs, only **forward model (−0.078)** and **servo
  (−0.074)** lower the cosine. Efference, sensor and privileged critic each lower it by
  ≤ 0.014.
- So the forward model and the servo change *how* the policy is solved. The other three
  factors change it about as much as each other, and less.

## 4. Inferring the factors from weights (leave-one-seed-out, balanced accuracy, 300 permutations)
Shape-shared layers only (actor 1–4 + head, critic 1–5; 105 features that reveal nothing
about the architecture). The permutation null's 95th percentile is 0.61–0.66 throughout;
every p below is at the permutation floor (0.003–0.005) unless stated.

| factor | all 62 | all, reward-residualised | within subset, reward-residualised (actor layers only) |
|---|---|---|---|
| efference | 0.90 | 0.62 (p=0.08) | MLP 1.00, FM 0.83 |
| forward model | 1.00 | 0.57 (p=0.22) | 1.00 (eff+priv runs) |
| sensor | 0.98 | 0.92 | MLP 0.76, FM 0.75 (p=0.02) |
| privileged critic | 1.00 | 0.60 (p=0.14) | MLP 0.86 |
| servo | 1.00 | **1.00** | MLP 1.00, FM 1.00 |

- The weight features predict **reward itself** with leave-one-seed-out R² = 0.85. That is
  why the pooled classifier can lean on reward, and why the residualised column matters.
- After residualising on reward, the pooled linear classifier loses efference, forward model
  and privileged critic. Within a network type they come back (0.8–1.0), so their signatures
  differ in MLP and FM networks rather than being absent.
- **Servo is the one factor written plainly into the policy's weights.** It is inferable at
  1.00 in every set, including actor layers only with reward removed.
- **Sensor information sits mostly in the critic** (best single feature: critic1 bias std,
  0.92). From the actor alone it drops to 0.75–0.76.
- **Matched-reward, same-architecture servo pairs** (eff+fm+priv vs eff+fm+servo+priv, both
  ≈835, seeds 43–45; eff+fm+sensor+priv vs eff+fm+servo+sensor+priv, ≈909/917, seeds
  44–46): 6 same-seed pairs.
  - Leave-one-pair-out "which one is the servo" is **6/6 correct**.
  - Their ΔW cosine is 0.18, against 0.53 for replicates.
  - Features that flip in the same direction in all 6 pairs (sign test p = 0.03 each):
    - predictor efference lag-1 row ↑ (+2.9 sd) and lag-2 row ↑
    - predictor velocity rows ↓ (−1.8 sd)
    - predictor foot-orientation rows ↓
    - head knee std-column norm ↑
    - actor3 norm and relative change ↓
    - head right-knee mean-bias ↓
- **Best single features** (leave-one-seed-out):
  - forward model: the first-layer row ratios (1.00; trivially so, since the predictor has
    its own loss)
  - privileged critic: critic2 relative change (0.96; lower for a privileged critic)
  - servo: critic1 ΔW effective rank (0.95) and actor3 relative change (0.93)
  - sensor: critic1 bias std (0.92)
  - efference: no single feature is good (best 0.77, first-layer qvel.vz row)

## Caveats
- n = 62 across 24 unbalanced cells (several have n = 1) and 5 init seeds. Leave-one-seed-out
  has 5 folds, so accuracy is coarse. Permutation p is floored at 1/301 or 1/201.
- The factors are nested in the design (FM ⇒ privileged critic; the delayed critic only
  exists in MLP runs). Pooled accuracies for those two can ride on each other, which is
  why the subset rows exist.
- The reward residualisation is linear and global. It is a sanity check, not a matched
  design.
- Row-norm ratios compare to init, so they are fan-in-free. Efference inputs are raw
  actions in [−1, 1], not normalised, so their absolute scale is not directly comparable
  with the normalised observation rows; the ratio-to-init still is.
- The replicate floor comes from 8 pairs, 4 of which include a 2 G run read at its 480 M
  checkpoint.
