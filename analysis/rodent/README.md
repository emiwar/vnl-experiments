# Rodent analyses

Imitation of motion-capture clips by the simulated rodent, in
`emiwar-team/nnx-ppo-rodent-delays`. The shared pipeline — run index, artifact store, the
question folder, the comparability protocol, the plotting conventions — is
[`../README.md`](../README.md); this file holds what is true of *this track only*, and the
traps that have already produced a wrong or nearly-wrong conclusion here.

| | |
|---|---|
| WandB project | `emiwar-team/nnx-ppo-rodent-delays` (the `index.DEFAULT_PROJECT`) |
| Run index | [`../_runs/nnx-ppo-rodent-delays.jsonl`](../_runs/nnx-ppo-rodent-delays.jsonl) |
| Template | `cp -r ../_template/rodent analysis/rodent/<question-slug>` |
| Control step | `ctrl_dt = 0.01 s` — 1 step = **10 ms**, the `style.CTRL_DT_MS` default |
| Cohort tag | `TrainEvalSplit` — asserts the run held out a clip split |

### The three eval datasets

**`train`** is the 80 % training split (the clips the policy trained on); **`old_eval`**
the held-out 20 % (unseen clips, same 250-frame length); **`new_eval`** 32 fresh
1500-frame clips. Each record carries, per dataset, `episode_reward`, `lifespan_steps`,
per-reason `termination_rate` (incl. `survived`), per-step `errors` and network
`net_metrics` (e.g. `fm_pred_mse`), plus hierarchical `param_counts`.

Name the dataset in every axis label and every reported number — `reward_label()` in
`wandb_utils.style` exists for exactly that, and §7 of the shared README explains why.

## Traps

Read before starting. The general ones — nondeterminism, GPU confounds, stale artifacts,
`scan_history` — are in [`../README.md` §6](../README.md#6-traps); these are the ones
peculiar to the rodent task.

> Scripts in this track cite traps as "`analysis/README.md` §6", which is where all of
> them lived until the 2026-09-10 track split. If a citation names something you cannot
> find there — `body_target_frame`, the walker XML, the unregularised runs, `noproprio`
> delays, the eval datasets — it is on this page. Both lists are mandatory; only the
> filing changed.

**`body_target_frame` lives on the env, not the network.** `AbsoluteImitation` reads
`config["env_params"]["body_target_frame"]`. The copy under
`config["net_params"]["body_target_frame"]` is **inert** — the training scripts set it on
`net_config`, where nothing reads it, and it was only logged. Every AbsoluteImitation run
before 2026-07-06 therefore trained with `current_root` regardless of what `net_params`
shows. Always filter and label on `env_params`. This invalidated the reference-root vs
current-root comparison in [`imitation-target-representation/`](imitation-target-representation/).

**A decoder-input ablation looks exactly like the standard efference baseline.**
`net_params.dec_use_intention=False` / `dec_use_proprioception=False` (added 2026-08-21)
drop the intention or the proprioception stream from the enc-dec decoder. Such a run keeps
the standard `{enc,dec,critic}_hidden_sizes`, keeps `efference_length == delay_k`, and
carries the `TrainEvalSplit` tag — so it passes every test the "standard efference
baseline" cohorts apply and joins them silently. Every folder that selects that cohort now
gates on `pipeline.full_decoder_inputs(net)` (dict form, for the live-fetch extractors) or
`& pipeline.full_decoder_inputs_mask(df)` (index form). Use those rather than a plain
filter kwarg: the columns are **absent**, not `True`, on every run predating the flags, so
`index.select(..., **{"net_params.dec_use_intention": True})` would drop the entire history.
The run name and tags also carry a `nointent` / `noproprio` token.

**`body_target_frame="reference_root"` does not make the imitation target
state-independent** — use `reference_root_open_loop` if that is what you want. `reference_root`
selects the egocentric frame for the `body` sub-key of `task_obs` and nothing else. `root` and
`quat` are computed above that branch and are, under both `current_root` and `reference_root`,
the reference root pose relative to the **current** root — an undelayed root position and
orientation error, 35 of the 640 `task_obs` numbers, in the units of the `root_too_far` (0.1 m)
and `root_too_rotated` (60°) terminations. The config docstring used to say "the pure target
pose shape, independent of all current state", which describes the body targets it was about
and was read as a claim about the whole target.

This matters for any experiment that delays or ablates the proprioception stream and then
calls the result feedforward: `task_obs` is neither delayed nor ablated by those knobs. It
surfaced in [`position-control-open-loop/`](position-control-open-loop/), whose `frame_leak.py`
measures which sub-keys move when the walker is displaced (and confirms the identity
`root[0] == rotate(ref_root_pos - root_pos, root_quat)`).

**The third value, `reference_root_open_loop`** (added 2026-09-02) anchors `root`/`quat` to the
*reference* root at the current frame instead, making the whole target a function of
`(clip, current frame)` with no dependence on the walker. Three things about it are worth
knowing before selecting on it:

* It is a **value, not a new key**, precisely so that cohorts stay safe: every committed
  analysis selects the frame by equality (`== "reference_root"`), so all of them exclude
  open-loop runs without being touched. A separate key would have been the
  `dec_use_intention` / `dec_use_proprioception` trap again — invisible to existing selectors,
  and needing a `full_decoder_inputs_mask`-style retrofit in every folder.
* The default stays `current_root`, so every stored `config.json` reloads to the behaviour it
  trained with, and the two existing values are numerically untouched (`frame_leak.txt`
  rebuilds byte-identically across the change).
* Its runs carry an `openloop` **tag** — not a name token, since this is expected to become
  the default and a token would then lengthen every run name for nothing. Filter on the tag,
  or on `env_params.body_target_frame`. The offline env-class resolvers key off
  `absolute_imitation.BODY_TARGET_FRAMES` rather than a local list — `eval_videos` used to
  hardcode the two old values and would have silently rebuilt an open-loop run as the base
  `Imitation` env, whose targets are *relative* to the current state.
* **It also takes the root error away from the critic.** `build_delay_network` feeds the critic
  undelayed `task_obs + proprioception`, and proprioception has no root-position error, so the
  value function loses its most direct predictor of a `root_too_far` termination. Expect the
  critic to be worse, not just the actor, and do not read a reward drop as purely an actor
  effect.

**A `noproprio` run's `delay_k` is inert, and its name says otherwise.** `build_delay_network`
puts the `Delay` layer *inside* the proprioception branch, and `dec_use_proprioception=False`
does not construct that branch — so `delay_k` reaches nothing. `RodentEncDec_delay10_eff10_noproprio`
is not a delay-10 experiment: at fixed `efference_length` the two networks are
bit-identical (same parameter count, same weight and carry trees, `max|Δ output| = 0`), which
`position-control-open-loop/check_delay_inert.py` asserts. What actually varies across those
runs is `efference_length`, which the launcher ties to `delay_k` by default, so plotting them
against delay silently plots the efference queue length with the wrong axis label —
and `delay0_eff0_noproprio` vs `delay5_eff0_noproprio` is a replicate pair, useful as a
noise floor (3.1 % on held-out reward) rather than a delay contrast. The same applies to any
future ablation that removes the branch a manipulation lives in: check what the *built*
network depends on, not what the config records.

**Every offline rebuild used to silently swap the walker XML** (fixed 2026-08-18). A
checkpoint records the *cluster* path of its XML, which does not exist on a laptop, so
`parse_env_config` repaired it — by taking the local default, `consts.RODENT_XML_PATH` =
`rodent.xml`. A run trained on `rodent_no_tail_collisions.xml` was therefore re-simulated on
a **different body**, in three independent copies of the same block
(`delays/evaluation.py`, `delays/eval_videos.py`, `tools/checkpoint_utils.py`) and so in
every `eval`, `activations` and `video` artifact of the new-XML cohort. The inline
end-of-training eval was unaffected — it is handed the live `env_config` — which is exactly
why the discrepancy hid: offline `old_eval` reward sat 2 % / 13 % / 27 % / 42 % below inline
at delays 0 / 10 / 20 / 50, with survival 0.23 vs 0.67 at delay 50, and nothing in the
artifact said which body it had used.

Three lasting consequences:

* `envs/config_io.resolve_local_xml_paths` now repairs the *directory* while keeping the
  *file*, and warns loudly when the run's own XML is genuinely unavailable locally.
* Producers stamp `resolved.walker_xml_path` / `arena_xml_path`. **An absent stamp means
  pre-fix**, which is what makes the damage decidable after the fact.
* `artifacts audit-env` classifies every stored artifact as broken / adoptable / repaired
  and writes the re-production run lists. It found 363 eval, 16 activation and 16 video
  artifacts built on the wrong body; the analyses resting on them were
  [`action-noise-robustness/`](action-noise-robustness/) (all six eval specs),
  [`collision-model-xml/`](collision-model-xml/) (67/149 runs) and
  [`xml-ceiling-vs-convergence/`](xml-ceiling-vs-convergence/) (16/29). It reports zero once a
  folder is repointed, so it doubles as the done-check — it reads *pinned* spec ids only, not
  every id a script mentions, so a retirement note in a comment does not keep a fixed folder
  flagged.

The general lesson is narrower than "check your paths": **a reconstruction that repairs an
input must record what it chose.** Any field a rebuild silently substitutes is a field no
downstream analysis can audit.

A second lesson, from how long this survived: **verifying provenance against the record that
was already trusted is not verification.** `collision-model-xml/report.md` stated that "every
eval was verified to have used the run's own body and frame" — and it had been, by comparing
the run's stored config against itself. Nothing in that check could see what the eval process
actually loaded. Post-fix, the equivalent check reads `resolved.walker_xml_path` off the
artifact and compares it to `env_params.walker_xml_path` from the index — two independently
written records. `assert_artifact_body` in `collision-model-xml/extract.py` and
`xml-ceiling-vs-convergence/extract.py` is the pattern; copy it into any folder that spans two
bodies.

**Outcome, 2026-08-19.** [`collision-model-xml/`](collision-model-xml/) carries a retraction:
its headline result ("the new body falls over far more often; survival collapses") was the
artefact, and the corrected data reverses it. The v1 evals understated new-XML held-out reward
by 3.8 % at delay 0 rising to 77 % at delays 90–100, while the 79 old-XML runs are
bit-identical across the fix — so the bug acted on exactly one arm of every contrast in that
folder. [`xml-ceiling-vs-convergence/`](xml-ceiling-vs-convergence/) went the other way: its
curve-based primary result never moved (curves come from `history`), and the newly-usable evals
gave it a held-out measurement it had been missing on 25 of 29 runs.

**The in-training `eval/*` series measured the *train* split** (fixed 2026-08-20). Both
delays training scripts built a held-out env and then threw it away:

```python
eval_env = AbsoluteImitation(env_config, clips=test_clips)
eval_env = train_env          # <- train_rodent_delays.py:173, train_rodent_forward_model.py:188
```

(Those scripts were replaced by `vnl_experiments/train.py` in the 2026-09-01 Hydra
migration; the line references are to their last state in git history. The corrected
construction, and a comment saying not to re-simplify it, live in `build_run` there.)

The override dates from the file's creation (`e5bbf3f`, 2026-06-01), so **every delays run
to date** reported in-training eval on the clips it was training on. What that touches:

* **Affected: the WandB `eval/*` metrics** — `eval/episode_reward/*`, `eval/lifespan/*`,
  `eval/net/*`, `eval/env/*`. These are what the `history` artifact captures, so any
  curve-based result is a train-split learning curve, not a generalisation measurement.
* **Not affected: the offline `eval` artifacts, nor the inline end-of-training eval**
  (`evaluation.run_final_eval`). Both are handed `train_clips` / `test_clips` explicitly and
  build their own envs via `build_datasets`, so their `train` / `old_eval` / `new_eval`
  datasets were always genuinely separated. Held-out numbers taken from the store are sound.

The comparability consequence belongs with §4: `eval/*` means *train-split* before the fix
and *held-out* after it. **Never pool or contrast `eval/*` across 2026-08-20** — a post-fix
run will look worse for reasons that have nothing to do with what it is testing. Add
`created_at` (or `git_commit`) to `INVARIANTS` in any folder whose runs straddle the date,
and prefer the offline `eval` artifacts for anything held-out.

Action: audit any question folder whose primary result rests on `history`-derived `eval/*`
curves. [`xml-ceiling-vs-convergence/`](xml-ceiling-vs-convergence/) is the first to check —
§6 already records that its primary result is curve-based from `history` — and the check is
whether "held-out" is claimed anywhere for a number that came from a curve. The fix itself
changes no stored artifact and needs no `VERSION` bump.

**19 runs trained with the regularisation switched off** (2026-08-21 to 2026-08-24).
`delays.network_builders._parse_net_params` ran `int(v)` on every value, and
`int(0.01) == 0` in Python, so every sub-1.0 float was truncated to zero. The parser
existed to decode the stringified `config.json` on the **eval** side, where those four
values only touch a regularisation term and a reported metric. The 2026-08-21 registry
refactor put **training** on the same path, and from then until the fix every run trained
with:

| net_param | config says | actually used |
|---|---|---|
| `entropy_weight` | 0.01 | **0** — no entropy bonus |
| `kl_weight` | 0.001 | **0** — no KL penalty |
| `min_std` | 0.1 | **0** — no policy-std floor |
| `latent_min_std` | 0.01 | **0** |

The result is premature entropy collapse: on `old_eval` the action std fell 0.167 -> 0.049
(below the floor that was supposed to be enforced — that impossibility is what exposed the
bug), the bottleneck KL rose 21.7 -> 146.0, `root_too_far` terminations went 1.2 % -> 18.3 %,
and reward fell **14 % at delay 0 and 36 % at delay 10** against the matched 2026-08-11
runs. It hits every architecture equally, so the affected LSTM runs say nothing about
recurrence.

**The logged config does not show it.** `net_params` records the intended 0.01 / 0.001 /
0.1 — the truncation happens after the config is written — so no column distinguishes a
broken run and no comparability report can flag one. The discriminator is the commit:
`pipeline.UNREGULARIZED_COMMITS`, with `pipeline.regularized_training_mask(df)` as the row
mask. `created_at` is **not** a safe proxy: `dgmexgcj` trained on 2026-08-21 from the
earlier `afbeea0` and is fine.

The general lesson is the sibling of the walker-XML one: **a value that is recorded before
it is transformed is not a record of what ran.** `env_params` is authoritative because the
env reads it directly; `net_params` was not, because a parser sat between it and the
network. Where a config passes through coercion, the thing worth asserting is the value the
*module* ended up with — which is what `network_builders_test.ParseNetParamsTest` now does.

`eval`, `activations` and `video` went to `VERSION = 3` on 2026-08-24, because the fix
restores `latent_min_std` and the bottleneck samples at eval time, shifting the actions
slightly (mean |delta| 5.4e-4, max 1.2e-2; critic values unchanged). Old artifacts stay
valid at their pinned `spec_id`s and need re-producing only for a formal old-vs-new
comparison. `history` is unaffected and keeps `VERSION = 1`.

**Two clocks in the eval datasets.** `clip_length` is in **mocap frames @ 50 Hz** (250 →
5 s, 1500 → 30 s), but the policy runs at `ctrl_dt = 0.01 s` (100 Hz). Rollouts are
`ceil(frames / (ctrl_dt·mocap_hz)) + 2` control steps — 502 for train/`old_eval`, 3002 for
`new_eval`. `lifespan_steps` and delays are in control steps (1 step = 10 ms).

The three eval datasets are: **`train`**, the 80 % training split (the clips the policy
trained on); **`old_eval`**, the held-out 20 % split (unseen clips, same 250-frame length);
and **`new_eval`**, 32 fresh 1500-frame clips. Each record carries, per dataset,
`episode_reward`, `lifespan_steps`, per-reason `termination_rate` (incl. `survived`),
per-step `errors` and network `net_metrics` (e.g. `fm_pred_mse`), plus hierarchical
`param_counts`.

**Raw reward is comparable within a dataset, not across.** Cumulative `episode_reward`
scales with clip length (~6× on `new_eval`). Across datasets use `reward_per_step` and the
per-second `hazard_rate` = `(1 − survived) / mean-alive-time` (failure terminations only;
end-of-clip truncations are censored, not events). Prefer the hazard to a raw survival
fraction: survival penalises longer clips for having more chances to fail.

## The legacy directories

`eval_results/eval_results/` (263 batch eval JSONs), `eval_results/old_eval_results/` (an
older batch) and `eval_results/activations/` (13 files, 22 GB, keyed by run *name*) have
been adopted into the store by `artifacts import-legacy`, **hardlinked** so nothing was
duplicated. They appear under fixed, unhashed spec ids — `legacy-batch`, `legacy-batch-v0`,
`legacy-<dataset>` — because their true specs were never recorded and inventing a hash
would imply a precision that does not exist. The originals can be deleted once
`artifacts verify` is clean; `eval_runs.py --collect` still gathers inline `eval.json`
files into `eval_results/eval_results/`, so re-run `import-legacy` after a collect.

Analyses written before this pipeline (everything except
[`collision-model-xml/`](collision-model-xml/),
[`xml-ceiling-vs-convergence/`](xml-ceiling-vs-convergence/),
[`explicit-vs-implicit-fm-2g/`](explicit-vs-implicit-fm-2g/) and
[`explicit-vs-implicit-fm-budgets/`](explicit-vs-implicit-fm-budgets/)) still read
`eval_results/`
directly and fetch from WandB in their `extract.py`. They remain valid and their CSVs are
unchanged; convert one to the layout above when you next need to touch it.
