# Implicit forward model probe — notes (feeds report.md)

`probe.py` → `probe.csv`, `figures/probe_layers_{delta,current}.png`, `figures/probe_summary_{delta,current}.png`.
62 runs, 3 course sets × 3 episodes × 960 steps (first 40 dropped). Ridge, 3 folds = episode index; alpha by inner CV.

**Delta score definition.** Each representation is ridge-decoded to the *current* normalised obs; "delta" is that same
prediction error normalised by Var(obs_t − obs_{t−5}) instead of Var(obs_t). For the input, this is the same as
regressing the delta directly. For deep layers it doesn't require the layer to also carry the delayed obs. It is also
the scale on which the FM's own prediction (`pred_out_direct`, no regression) is scored. Regressing the delta directly
from a deep layer was tried first and dropped: `pred_out` scored 0.69 that way, against 0.93 when the prediction itself
was used.

## Family means, proprio dims 0–23, held-out R²

| family | n | input (linear) | input_rf (2048 random swish) | best actor layer, trained | same, init | FM best pred layer (tr / init) | FM explicit pred_out |
|---|---|---|---|---|---|---|---|
| **delta** |||||||
| MLP, no eff | 17 | 0.614 | 0.758 | 0.751 | 0.722 | – | – |
| MLP + eff | 21 | 0.691 | 0.791 | 0.767 | 0.755 | – | – |
| FM, no eff | 7 | 0.697 | 0.815 | 0.905 | 0.686 | 0.905 / 0.790 | 0.896 |
| FM + eff | 17 | 0.758 | 0.843 | 0.909 | 0.707 | 0.918 / 0.813 | 0.909 |
| **current** |||||||
| MLP, no eff | 17 | 0.598 | 0.731 | 0.729 | 0.697 | – | – |
| MLP + eff | 21 | 0.659 | 0.758 | 0.731 | 0.720 | – | – |
| FM, no eff | 7 | 0.624 | 0.759 | 0.867 | 0.623 | 0.863 / 0.729 | 0.850 |
| FM + eff | 17 | 0.716 | 0.816 | 0.879 | 0.671 | 0.882 / 0.779 | 0.867 |

Per-run gaps on delta, as mean [min, max]:

| family | trained − linear input | trained − init | ceiling − trained |
|---|---|---|---|
| MLP, no eff | +0.137 [0.11, 0.17] | +0.029 [0.016, 0.044] | +0.007 [−0.012, 0.019] |
| MLP + eff | +0.076 [0.05, 0.09] | +0.012 [−0.020, 0.025] | +0.024 [0.012, 0.036] |
| FM, no eff | +0.208 | +0.219 | −0.090 |
| FM + eff | +0.152 | +0.202 | −0.067 |

In every MLP run the best layer is act0–act2. In every FM run it is decoder act0, which carries the prediction
(act0 ≈ pred_out).

## Reading

- **No implicit forward model in the DelayedMLPs, with or without efference.**
  - Every MLP layer does beat the *linear*-input baseline (+0.08 to +0.14), but its untrained init does almost as well:
    trained minus init is only +0.01 to +0.03.
  - No MLP layer reaches the random-feature ceiling: it sits 0.01–0.04 above them in every MLP+eff run.
  - So the extra over the linear baseline is what any random nonlinear expansion of the input gives, not a learned
    state estimate. This is the rodent result (`../../rodent/implicit-forward-model/`) again, with a stronger control.
- **Efference doesn't change this.**
  - It raises the linear baseline (0.61 → 0.69), so recent actions do linearly carry information about the current state.
  - But trained minus init is *smaller* with efference (0.012 vs 0.029), so the MLP+eff nets use the queue without
    building a better state estimate from it.
- **The explicit FM does build one, and the decoder keeps it.**
  - Its prediction scores 0.90–0.91 on delta, above the random-feature ceiling (0.82–0.84) and far above init
    (0.69–0.71 for decoder layers).
  - Decoder act0 decodes the current state as well as pred_out does. The information then fades with depth
    (act3 ≈ 0.83–0.88).
  - FM without efference is almost as good (0.905). On flat ground the efference copy helps the FM (0.951 vs 0.937);
    near blocks it helps less consistently.
- **Contacts.**
  - Scores are lower on the block courses than on flat ground for every family. For the MLP families the gap is
    0.10–0.16 on the irregular course and 0.10–0.11 on stairs; for the FM families it is 0.06–0.08 and about 0.04.
  - Splitting the block courses into near-block (|blk0.dist| < 0.5 m) and far-from-block frames barely matters:
    explicit FM 0.881–0.893 near vs 0.897–0.910 far.
  - So the drop is spread over the whole course, not concentrated at the block edge.
- **Side observation: servo dynamics are more predictable.** The linear-input delta R² is 0.66 for torque and
  0.72–0.74 for servo, whatever the sensor setting. This raises every baseline in servo conditions.
- **The block-sensor dims are poorly predicted by everything.**
  - They jump when a block passes, and the MLPs score at or below 0.13 on them (sometimes negative).
  - The FM's decoder act0 reaches 0.55–0.58 (sensor runs only).

## Caveats

- This is linear decodability only.
- Cells with one run: fm+priv, fm+servo+priv, servo, servo+priv, eff, eff+servo, eff+servo+priv.
- Seeds are shared across conditions, so seed is the unit of replication. No seed-level statistics are done here.
- The init control uses the trained run's normaliser on its input; the init network was never run in closed loop.
- `input_rf` is one fixed draw of random features (scale 1.5/√p, 2048 units). It is a rough ceiling, not a bound.
