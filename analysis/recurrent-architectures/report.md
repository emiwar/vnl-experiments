# Recurrent decoders: architecture, delay tolerance, action buffer, BPTT horizon

## Question

Five questions on one cohort: is a recurrent decoder better than feedforward (with or
without an explicit forward model) *given its extra parameters*; how much delay it
tolerates; whether a longer BPTT horizon helps everyone; whether it can run on a one-step
action buffer and compensate internally, and out to what delay; and whether the cell type
matters.

## Dataset & comparability

One cohort: **4096 envs, seed 42, new XML, `reference_root`, full decoder inputs,
regularisation intact, trained to completion.** 65 runs.

| condition | n | delays covered | params |
|---|---|---|---|
| `feedforward` | 37 | 0–100 | 4.13–6.07 M |
| `forward_model` (canonical: `fm_loss_weight = 1`, detached) | 2 | 10 | 5.39 M |
| `lstm` | 22 | 0–60 | 6.24–7.39 M |
| `gru` | 1 | 10 | 5.89 M |
| `rnn` | 3 | 5, 10, 20 | 4.67–4.84 M |

Parameter count grows with `efference_length` (the queue widens the decoder input), so the
ranges above span the delay sweeps, not architecture variants.

**Comparability: no invariant varies** — `env`, `n_envs`, `seed`, `total_steps`,
`learning_rate`, `n_minibatches`, `latent_size`, encoder and critic sizes,
`body_target_frame`, `ctrl_dt`, `walker_xml_path`
([comparability.txt](comparability.txt)). `delay_k`, `efference_length` and
`rollout_length` are the axes and are carried as columns.

**Requeued jobs are handled, not hoped about.** The cluster terminated and requeued most
of these, leaving a partial and a complete twin per config. A run enters the cohort only
once `summary._step ≥ config.ppo.total_steps` — it trained to completion, not merely past
a threshold — so no partial can enter whatever fraction it reached. (The earlier
`MIN_STEPS = 590 M` admitted exactly the same 119 runs, but would have let through a
future partial that died at 595 M.)

`state` is deliberately **not** filtered. 23 of the pre-refactor (`ef060b7`) feedforward
runs are `state = failed` despite having trained the full 600 M steps and written a
`final_eval` summary — the job exited non-zero after the work was done. Filtering on
`state` would silently delete the backbone of the delay sweep.

Completion being guaranteed, a cell holding two runs is a real replicate rather than a
partial in disguise, so `extract.py` reports them instead of failing. Three exist, and
they measure different things:

| cell | kind | spread |
|---|---|---|
| feedforward, delay 0, eff 0, rollout 20 | cross-epoch (`ef060b7` / `a3450a9`) | 0.1 % |
| feedforward, delay 5, eff 5, rollout 20 | cross-epoch (`ef060b7` / `a3450a9`) | 4.6 % |
| LSTM, delay 40, eff 40, rollout 60 | **same-config** (`4245ae4` twice) | 1.1 % |

`plot.py` averages all three.

**Older forward-model runs are excluded.** They predate `fm_loss_weight` /
`detach_prediction` and span 1130–1998 reward at delay 10 across variants; only the two
canonical runs are used.

**Preliminary.** `metric_source` records artifact vs inline `final_eval` (+0.35 % ± 1.53 %
calibration). **Every other cell is n = 1**, against a replicate spread of ~2 % at delay 0
rising to ~8.5 % at delay 10 — differences below that are not resolved.

The LSTM delay-40 pair puts a floor under that. It shares commit, seed and every
hyperparameter, so its 1.1 % spread (990 vs 979) is *pure run-to-run nondeterminism* —
GPU kernel scheduling, not seed. So of the ~8.5 % seed spread at delay 10, only about a
percent is irreducible; the rest is genuine seed sensitivity.

## Figures

Architecture at the one cell every arm shares, with parameter counts on the bars and the
reward-vs-parameters view beside it.

![Architecture comparison at delay 10.](figures/architectures_delay10.png)

Delay tolerance, and whether a one-step action buffer suffices. The two dotted curves are
the matched `efference = 1` sweeps: they start together at delay 0 and diverge from there.

![Reward vs delay for each architecture and buffer length.](figures/delay_tolerance.png)

BPTT horizon, over every cell measured at both rollout 20 and 60.

![Reward vs rollout_length.](figures/rollout_length.png)

## Conclusions

**1. Recurrent beats feedforward and the explicit forward model, and the delay-0 control
now argues this is not just capacity.** At delay 10, efference 10, rollout 60:

| architecture | reward | vs feedforward | params |
|---|---|---|---|
| Vanilla RNN | 1490 | −9.2 % | 4.84 M |
| Feedforward | 1641 | — | 4.32 M |
| Explicit forward model | 1744 | +6.3 % | 5.39 M |
| GRU | 1846 | +12.5 % | 5.89 M |
| LSTM | 1912 | +16.5 % | 6.42 M |

Reward rises monotonically with parameters across four of the five arms, which is exactly
the confound the question anticipates. **The vanilla RNN is the one point that breaks it**:
it carries 12 % *more* parameters than the feedforward net and scores 9.2 % *less*.

The new `delay 0, eff 1, rollout 60` pair sharpens this considerably. There the LSTM
(6.24 M) scores **2167** and the feedforward net (4.14 M) scores **2149** — **+0.8 %**,
inside the ~2 % delay-0 replicate spread. So 2.1 M extra parameters and a hidden state buy
essentially *nothing* when there is no delayed information to integrate, and **+68 %** at
delay 10 under the identical buffer setting. What the recurrent decoder supplies scales
with the amount of missing feedback, not with capacity.

**Caveat:** delay 0 is close to the task ceiling (every arm lands at 2145–2167), so it is a
weak place for *any* architecture to demonstrate extra capacity. The argument is
suggestive, not conclusive, and a parameter-matched feedforward control at delay 10 remains
the top follow-up.

**2. Delay tolerance degrades gracefully to at least 60 steps (0.6 s), and the feedforward
net does not follow.** The LSTM with a one-step buffer runs 2167 → 2067 → 1831 → 1654 →
1290 → 1233 → 1110 → 954 → 941 → 897 across delays 0, 2, 7, 10, 15, 20, 30, 40, 50, 60.
There is no cliff inside the tested range; the curve is still falling smoothly at 60.

The matched feedforward sweep — now complete at rollout 60 — collapses instead:
2149 → 1845 → 1255 → 983 → 887 → 843 → 751 across delays 0, 2, 5, 10, 20, 30, 60. It is
essentially flat below ~900 from delay 10 onward, i.e. it has stopped extracting anything
from the delayed channel. The LSTM's margin over it peaks at delay 10 and then narrows as
both approach the floor:

| delay | 0 | 2 | 10 | 20 | 30 | 60 |
|---|---|---|---|---|---|---|
| LSTM ÷ feedforward (eff 1, rollout 60) | +0.8 % | +12.0 % | **+68.3 %** | +38.9 % | +31.8 % | +19.5 % |

The LSTM **full-buffer** sweep is now complete at rollout 60 too: 2125 → 2061 → 1912 →
1447 → 1209 → 985 → 759 across delays 2, 5, 10, 20, 30, 40, 60. Note the last point — it
falls *below* the same network's one-step curve, which is conclusion 4.

**3. A longer BPTT horizon helps only the recurrent decoder.** Seven cells are now measured
at both rollout 20 and 60:

| cell | rollout 20 → 60 |
|---|---|
| Feedforward, delay 5, eff 1 | +3.6 % |
| Feedforward, delay 5, eff 5 | −3.8 % |
| Feedforward, delay 10, eff 10 | −6.5 % |
| Feedforward, delay 20, eff 20 | −7.7 % |
| Forward model, delay 10, eff 10 | −0.8 % |
| LSTM, delay 10, eff 1 | **+9.5 %** |
| LSTM, delay 10, eff 10 | **+8.2 %** |

Both LSTM cells gain ~8–10 %; three of four feedforward cells lose ground and the fourth
gains 3.6 %, which is within noise. So this is not a generic PPO improvement that happened
to favour the LSTM — it is specific to having a hidden state to propagate gradients
through. That also makes conclusion 1 conservative: the feedforward arm is compared at the
rollout length that suits it *worse*.

**4. Yes — and past delay ~40 the explicit buffer stops being worth carrying at all.**
Cost of truncating the buffer from `delay` to 1, all at rollout 60:

| delay | 2 | 5 | 10 | 20 | 30 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| LSTM | −2.7 % | — | −13.5 % | −14.8 % | −8.2 % | −3.1 % | **+18.3 %** |
| Feedforward | — | −34.5 % | −40.1 % | −24.9 % | — | — | — |

The LSTM's penalty peaks around delay 20 and then closes, and **at delay 60 it reverses**:
the one-step network scores **897** against the full-sixty-step network's **759**. The
buffer is not merely redundant there, it is *harmful* — 1.15 M extra parameters (7.39 M vs
6.24 M) spent widening the decoder input with sixty stale actions the network then has to
learn to ignore, while the hidden state already carries the same information in compressed
form. The crossover sits between delay 40 (−3.1 %) and 60 (+18.3 %).

The feedforward net reaches the same place from the other direction: at delay 60 it scores
751 with one step (rollout 60) against 768 with all sixty (rollout 20) — a 2 % difference
across a 60× change in buffer length. And the *full-buffer* LSTM at delay 60 (759) is
statistically indistinguishable from the full-buffer feedforward net (768), i.e. the long
buffer erases the recurrent advantage entirely.

In absolute terms the LSTM with a *one-step* buffer matches the feedforward net holding the
*full* buffer at delay 10 (1654 vs 1641), beats it at delay 20 (1233 vs 1181), and pulls
further ahead at 40/50/60 (954/941/897 against 929/823/768).

**Practical reading:** for delays past ~20 steps, spend the parameters on the hidden state
rather than on the action queue. This is the most actionable result in the folder and it
rests on n = 1 at the crossover — the delay-40 and delay-60 `eff = delay` points should get
a second seed before it is leaned on.

**Caveat, now narrower:** the `eff = 1` and LSTM full-buffer curves are complete at rollout
60, but the *feedforward* full-buffer reference beyond delay 20 is still at rollout 20 (no
rollout-60 feedforward runs exist past delay 20). Per conclusion 3 that flatters the
feedforward reference, so the comparison is conservative — but it is still a gap.

**5. Gating matters; LSTM vs GRU does not.** The ungated vanilla RNN is the worst arm
tested, below even the feedforward baseline despite more parameters. The LSTM leads the GRU
by 3.5 % (1912 vs 1846), which is inside replicate spread — treat them as equivalent, and
note the GRU gets there with 8 % fewer parameters (5.89 vs 6.42 M). On current evidence the
GRU is the better default, but it rests on a single run.

## Key runs to add

Ordered by how much they change what can be claimed.

1. **Parameter-matched feedforward at delay 10, eff 10, rollout 60** (~6.4 M — e.g.
   `dec_hidden_sizes = [1024]×4`). Still the single most load-bearing run in this list: the
   delay-0 control weakens the capacity objection but does not kill it, because delay 0 is
   at the task ceiling.
2. **GRU `eff 1` delay sweep** (10, 20, 30, 40). The GRU is statistically tied with the
   LSTM at 8 % fewer parameters and has exactly one data point supporting that.
3. **Feedforward `eff = delay` at rollout 60 for delays 30–60**, removing the last rollout
   confound from conclusion 4.
4. **Seeds 43 and 44 on the delay-10 / eff-10 / rollout-60 cell** for feedforward, LSTM and
   GRU. Everything here is n = 1 and the LSTM–GRU gap is inside replicate spread.
5. **Forward model with `eff 1`** at delays 10 and 20 — does the explicit predictor also
   survive buffer truncation, or is that specific to a learned hidden state? Directly
   comparable to conclusion 4 and currently untested.
6. **Vanilla RNN at `eff 1`, delays 10 and 30**, to confirm the ungated result is general
   rather than specific to the `eff 10` cell.
7. **A second seed at the conclusion-4 crossover** — LSTM `eff = delay` at delays 40 and
   60. The buffer-becomes-harmful result is the folder's most actionable claim and the
   reversal rests on one run per point.
8. **LSTM `eff 1` at delay 5 and rollout 60** (the existing delay-5 `eff 1` point is
   rollout 20), closing the last gap in the one-step curve.
9. **A short-buffer sweep at long delay** — LSTM at delay 60 with `eff` 1, 5, 10, 20. If
   the buffer turns harmful somewhere between 1 and 60 steps, this locates the optimum
   and tells you whether the right answer is "no buffer" or "a short one".
