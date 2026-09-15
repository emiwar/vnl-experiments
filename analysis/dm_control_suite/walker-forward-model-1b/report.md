# WalkerWalk at 1e9 steps: how much delay does an explicit forward model buy, and how much faster does it learn?

## Question

On WalkerWalk, does replacing the implicit state estimate inside a delayed actor with an
**explicit** supervised forward model (a) raise the delay the policy can absorb, and
(b) shorten the training needed to solve the task — at a budget long enough that the
baseline is no longer obviously mid-training?

An answer is two delay sweeps in which the arms are matched in everything but that
factor: one of end-of-training reward, one of **time to criterion**. The sibling folder
[`../explicit-forward-model/`](../explicit-forward-model/) asked (a) at 4.8e8 steps and
had to report its WalkerWalk answer as "at this budget" only, because its baseline was
still climbing at the end of training. Its first follow-up was to run longer; this is
that follow-up, with the time-to-criterion readout the speed claim actually needs.

## Dataset & comparability

- **Source:** WandB `emiwar-team/nnx-ppo-delays`, selected by the `CONDITIONS` in
  `extract.py` and frozen in `runs.csv`. 32 runs, 2026-09-09 → 2026-09-14.
- **Conditions:** two arms, faceted by budget. `efference_length == delay_k` in **every**
  run, so both arms are given the same information and differ only in whether the state
  estimate is built explicitly.

| condition | budget | n | seeds | delays | network | commits |
|---|---|---|---|---|---|---|
| `delayed_mlp` | 1e9 (primary) | 10 | 43, 46 | 0, 5, 10, 15, 20, 25 | `DelayedMLP`: actor on delayed obs + efference copy; privileged critic | `9e216ae` (7), `f9c8960` (3) |
| `flat_forward_model` | 1e9 (primary) | 8 | 43, 46 | 0, 10, 15, 20, 25 | `FlatForwardModel`: as above + supervised predictor (delayed obs, action buffer) → current obs, `fm_loss_weight = 1`, `detach_prediction = True` | `9e216ae` (8) |
| `delayed_mlp` | 4.8e8 (support) | 6 | 42 | 0, 5, 7, 10, 15, 20 | as above | `971ab99` |
| `flat_forward_model` | 4.8e8 (support) | 8 | 42 | 0, 2, 5, 7, 10, 15, 20 | as above | `971ab99` |

  `extract.py` asserts the arm-defining `net_params` per condition
  (`fm_loss_weight` / `detach_prediction` / `predictor_hidden_sizes` present exactly in
  the forward-model arm and absent in the MLP arm), so a mislabelled condition is an
  error rather than a caveat.

- **Reward reported:** the in-training eval series `eval/episode_reward/mean` — **unscaled
  and on full-length episodes**, i.e. every run is on the post-`971ab99` side of the
  eval-env fix (`reward_source = dmc_eval` for all 32). Note what this is *not*: there is
  no clip split in this track, so an eval episode is a fresh episode of the **same**
  distribution the policy trained on. Nothing here is held out, and nothing here is a
  generalisation result.
  - `reward_final` is the mean of the eval points in the last 5e7 steps — 84 points in
    every run, not the final point, which moves by a few percent (README §6).
  - `steps_to_900` is the first step at which the **trailing 1e7-step mean** reaches 900.
  - **Every y-axis is raw reward or raw steps.** Nothing is normalised, and the task
    maximum (1e3 by construction) is drawn as a reference line rather than divided
    through.
- **Task:** WalkerWalk only, so no across-task pooling arises. `ctrl_dt = 0.025` in all
  32 runs, so one delay step is 25 ms and the sweep spans 0–625 ms; `plot.py` asserts this
  against `data.csv` rather than trusting `add_ms_axis`'s rodent default of 10 ms.
- **Artifacts used:** `REQUIRES = ["index", "history:hist2000-8b97281e"]` — **32/32, no
  gaps** (`coverage.txt`). The non-default spec id is deliberate: the `history` producer
  takes the WandB project as a spec field defaulting to the *rodent* project, so these
  were produced with `--set project='"emiwar-team/nnx-ppo-delays"'`, and because the
  project is hashed into the spec, control-suite and rodent curves cannot be pooled by
  accident. It is also the *same* id the sibling folder used, so the 4.8e8 curves here are
  literally the same bytes that analysis read. `extract.py` additionally checks each
  curve's last step against the run's own `summary._step` and exits non-zero on a
  mid-training `history` snapshot (README §6); all 32 reach 1.000243e9 / 4.80215e8.

### Comparability verdict

**Programmatic** (`comparability.txt`, one section per budget plus a pooled section).
Within each budget and each arm: identical `n_envs`, `rollout_length`, `learning_rate`,
`n_epochs`, `n_minibatches`, `clip_range`, `discounting_factor`, `gae_lambda`,
`reward_scale`, `episode_length`, `ctrl_dt`, `sim_dt`, `impl`, `min_std`,
`entropy_weight`, `actor_hidden_sizes`, `critic_hidden_sizes`, `normalize_obs`,
`activation`, `initializer_scale`, all three eval settings, `vnl_playground` commit, and
`summary._step` (1.000243e9 exactly in all 18; 4.80215e8 exactly in all 14 — no run was
preempted short). The pooled section flags exactly `total_steps` and `summary._step` and
nothing else, which is the check that budget is the only axis separating the two cohorts.

Three columns are flagged, all about *code identity* rather than configuration:

| flagged | where | resolution |
|---|---|---|
| `git_commit` | inside the 1e9 **MLP** arm: 3 of 10 runs (delays 10/20/25, seed 43) at `f9c8960`, the rest and all 8 forward-model runs at `9e216ae` | see below — measured inert |
| `repos.nnx_ppo.commit` | the same 3 runs at `1725d20`, the rest at `5314279` | diagnostics-only diff |
| `repos.vnl_experiments.dirty` | `True` on 27 of 32 runs | voids the hashes, so the hashes are not the argument |

This is the one thing in the cohort worth taking seriously: at delays 10/20/25 of seed 43,
commit and seed coincide, so configuration alone cannot separate them. Three checks,
in increasing order of strength:

1. **`f9c8960`** adds `train.video.render_kwargs.camera: side` to six env configs. Render
   path only; it cannot reach the physics or the loss.
2. **nnx-ppo `1725d20 → 5314279`** ("Add checks for (and optional logging of) NaNs during
   the training") adds non-finite counters and a `check_diagnostics` that raises.
   `optimizer.update` is called identically and the update math is untouched — I read the
   `ppo.py` and `types.py` diffs directly. A side effect worth having: a run that
   *completed* 1e9 steps under `5314279` thereby proves it saw no non-finite reward,
   observation, action or gradient.
3. **`9e216ae`** adds `NaNGuardWrapper` to the training env stack, and this one genuinely
   can change a trajectory — but only on a step where MJX has already diverged.
   [`nan_guard_inert.py`](nan_guard_inert.py) tests whether that ever happened, using the
   fact that the guard produces a *termination* while a healthy WalkerWalk episode can
   only ever be *truncated*: `max(done_rate − truncation_rate) = 0.000e+00` across every
   logged iteration of **all 32 runs** ([`nan_guard_inert.txt`](nan_guard_inert.txt)). No
   run ever diverged, so the guard never fired, and the three stacks are functionally
   identical **on these runs**. That is a stronger statement than a clean commit hash, and
   unlike a hash it survives `dirty = True`.

The 4.8e8 cohort is at a single vnl-experiments commit and a single nnx-ppo commit across
**both** arms, so it is also a single-stack replication of the contrast.

**Manual.** I inspected the run configs directly rather than trusting tags or names (the
`min_std` sweep in this project is named identically to the defaults, and is gated out);
read the run names and the two selection gates; read the three commit diffs as above; and
confirmed the three shared-code repos. GPU model is mixed **within both arms at both
budgets** (1e9: MLP 4×A100 + 6×H200, FM 2×A100 + 6×H200), so it is not confounded with the
contrast — it would still confound a throughput comparison, which this analysis does not
make.

### Caveats

- **Two seeds at most, and one at delay 15.** Every 1e9 cell is n = 1 or 2 (delay 0 and 15
  are n = 1 per arm); the 4.8e8 cohort is n = 1 throughout. Treat the margins as effect
  *directions* with a rough size, not as measured effect sizes. The spread is real and
  visible: the two forward-model runs at delay 25 land at 946 and 755.
- **The MLP has not converged at delay ≥ 15 even at 1e9 steps.** Over the last 2e8 steps
  the MLP arm still gains +17 (delay 15), +7…+19 (delay 20) and +22…+24 (delay 25) reward,
  while the forward model is flat everywhere it solved the task (+0.5…+2.6). So the
  reward-vs-delay panel is *still* an at-this-budget statement for the MLP, just at a
  budget 2.1× longer than the sibling folder's.
- **Learning here is a phase transition, not a ramp** (see the training curves), so
  time-to-criterion is really time-to-transition, which is a high-variance quantity, and
  extrapolating the MLP's residual slope linearly is not sound. The MLP at delay 10
  transitioned at ~6.2e8 steps — *after* the point where the sibling analysis stopped
  looking — so a later transition at delay 15–25 cannot be ruled out by this budget
  either.
- **No forward-model run at delay 5 in the 1e9 cohort**, in either seed. The 4.8e8 panel
  covers it (FM 974 vs MLP 967, and 60 vs 74 M steps to criterion), which is why that
  cohort is kept; the 1e9 forward-model line simply skips delay 5.
- **The 900 criterion is this project's convention, not a published standard.** dm_control
  returns are bounded in [0, 1000] by construction and the benchmark literature reports
  curves rather than declaring a task solved, so there is no number to inherit. 900 sits
  clearly above where these runs plateau when they fail (515–755) and clearly below where
  they plateau when they succeed (946–981), and both arms reach ~980 at delay 0, so it is
  not a bar the architecture question itself sets. The sensitivity figure shows the
  ordering at 800 and 950 as well.
- **The two budgets are never averaged together**, only faceted. The MLP is the slower arm,
  so a shared 4.8e8 readout is biased towards the conclusion being tested: the MLP at
  delay 10 reads 695 at 4.8e8 and 944 at 1e9, because its curve steps up at ~6.2e8, past
  the shorter budget. The arm gap at delay 10 is consequently +271 at 4.8e8 against +26 at
  1e9 — an order of magnitude, nearly all of it budget rather than architecture. (Those
  two gaps are also different seeds, so treat the factor as indicative; the MLP's own
  695 → 944 is the unambiguous part.)
- **Single-seed noise in the 4.8e8 panel is large enough to break monotonicity**: its MLP
  scores 408 at delay 15 and 540 at delay 20. Read that panel for direction, not shape.

## Figures

![End-of-training reward and time-to-criterion vs delay, 1e9 cohort](figures/reward_and_steps_vs_delay.png)

The main result. **Left:** the two arms are indistinguishable to delay 5 (980 vs 980 at
delay 0), then separate sharply — FM − MLP is +26 at delay 10, +214 at 15, +376 at 20 and
+302 at 25, on a scale whose maximum is 1000. The forward model is still at 947–960 at
delay 20 (500 ms) where the MLP has fallen to 555–599. **Right:** the same contrast as a
learning cost. The MLP reaches the criterion in 25 M steps at delay 0 and 59 M at delay
5 — *faster* than the forward model's 47 M at delay 0, which is the price of the extra
predictor when there is no delay to compensate for — then needs 618 M at delay 10 and
never gets there at all by 1e9 steps for delay 15, 20 or 25 (open markers, one per run).
The forward model climbs gently instead: 161, 180, 289 and 504 M steps at delays 10, 15,
20 and 25. Note the broken forward-model line at delay 25: one of its two seeds is
censored, and a mean over the seed that happened to succeed is the one number this panel
must not show.

![The same two panels at the 4.8e8 budget, seed 42](figures/budget_480m_replication.png)

The independent replication, on a third seed and a single code stack, and the only place a
forward-model run at delay 5 exists. Same direction everywhere, with a crossover between
delay 7 and 10 rather than 10 and 15 — one seed earlier than the 1e9 cohort's, which is
about the size of the seed spread. Its delay-5 and delay-7 columns are the ones worth
taking from here (FM +7 and +21 reward; 60 vs 74 and 64 vs 218 M steps to criterion).
Do **not** read its levels against the panel above: its MLP is 2.1× short of the budget
that panel uses, which is most of why its delay-10 gap (+271) is ten times the 1e9
cohort's (+26).

![Every run's eval series at delays 10–25, 1e9 cohort](figures/training_curves.png)

The mechanism behind both readouts, and the figure that sets the error bars on the
conclusion. Learning on this task is a **step transition** — the walker finds gait and
jumps from ~600 to ~950 within a few tens of millions of steps — so "time to solve" means
"time until the transition", and the architecture's effect is on *when that arrives*: the
forward model transitions at 0.13e9 (delay 10) to 0.45e9 (delay 25), the MLP at 0.62e9
(delay 10) and not at all past that. Two things to look at directly: the MLP's delay-10
transition lands **just past** the dotted 4.8e8 line, which is exactly why the sibling
analysis could not settle this; and in the delay-15/20/25 panels the orange curves are
still visibly rising at 1e9, while the green ones have been flat for half the run.

![Time to criterion at 800, 900 and 950](figures/threshold_sensitivity.png)

The criterion is arbitrary, so this checks the claim does not live on it. At all three
bars the picture is the same: below delay ~5 the MLP is at least as fast, from delay 10 on
the forward model is several-fold faster, and past delay 10 the MLP is censored while the
forward model is not. What does move with the bar is only how many forward-model cells
survive at the far end.

## Tentative conclusion

**On WalkerWalk the explicit forward model clearly buys delay tolerance, and the
mechanism is speed of learning — but the sibling folder's "at this budget" caveat has been
narrowed, not removed.**

What the data shows:

- Up to delay ~5 (125 ms) the architectures are indistinguishable in final reward, and
  the forward model is if anything slightly *slower* to the criterion (47 vs 25 M steps
  at delay 0) — the predictor is overhead when there is nothing to predict around.
- From delay 10 (250 ms) on, the forward model dominates on both readouts, and the margin
  grows with delay: at delay 20 it holds 947–960 against 555–599, and it reaches the
  criterion in 289 M steps where the MLP does not reach it in 1e9.
- The forward model has converged at every delay it solved (flat over the last 2e8 steps);
  the MLP has not, at any delay ≥ 15.

What it suggests but does not establish:

- That the forward model reaches somewhere the MLP **cannot**. The MLP is still improving
  at 1e9 steps at delay ≥ 15, and its own delay-10 curve shows a transition arriving as
  late as 6.2e8 steps — so a longer budget could still move the delay-15/20/25 answer.
  What is now established is the weaker and more useful claim: **the forward model gets
  there several times sooner, and at delay ≥ 15 the MLP does not get there within 1e9
  steps at all.**
- That the delay-25 result is an architecture effect rather than a seed effect. One of the
  two forward-model runs there stalls at 755 and is still climbing, i.e. it looks like a
  run whose transition had not yet arrived — the same failure mode as the MLP's, just
  later.

Both locomotion readouts rest on two seeds per cell, so none of the margins above should
be quoted as an effect size.

## Follow-ups

- **Seeds at delay 15 and 25 before anything else.** Delay 15 is n = 1 per arm and carries
  the largest clean reward gap (+214); delay 25 is where the two forward-model seeds
  disagree by 191. Two or three more seeds in each would turn the headline from a strong
  hint into a result, and would put a spread on the time-to-criterion axis where it
  currently has none.
- **The forward-model run at delay 5, at 1e9**, to close the one hole in the primary grid
  and pin down where the crossover is. The two cohorts currently bracket it between 7 and
  15.
- **Does the MLP ever transition at delay 15?** A single MLP run at delay 15 taken to
  3e9–4e9 steps would separate "the forward model is faster" from "the MLP cannot" — the
  one question this budget still cannot answer. Its residual slope (+17 per 2e8) makes a
  naive prediction of ~1.8e9 more steps, but the transition structure of these curves
  means that number should be treated as an order of magnitude, not an estimate.
- **Report the predictor's own loss against the delay.** If the forward model's advantage
  is that it transitions sooner, the natural next question is whether the predictor's
  accuracy at the moment of transition is what gates it. `fm_loss` is logged but is not in
  the pinned `history` spec, so this needs a spec change (and therefore a new
  `HISTORY_SPEC_ID`) rather than a re-read.
- **Is the phase transition the same event in both arms?** A video or a gait readout at
  either side of the transition would say whether the two architectures converge on the
  same gait, which the reward number cannot distinguish.
- `FlatRecurrent` is a third architecture on this axis and is deliberately out of scope
  here; it is the obvious third arm for the same two panels.

---

*Reproduce:* `../.venv/bin/python analysis/dm_control_suite/walker-forward-model-1b/extract.py && ../.venv/bin/python analysis/dm_control_suite/walker-forward-model-1b/plot.py`
(add `--sync --refresh` to the extract to pull in runs added since `runs.csv` was frozen;
`--check` to verify the committed CSVs rebuild identically). The comparability check
`nan_guard_inert.py` needs network and WandB auth and is not part of the rebuild.
