# Analysis pipeline

This folder holds **reproducible WandB analyses**. Each scientific question gets its
own folder. The goal: a figure is reproducible from a committed CSV snapshot even
after more runs are added to WandB, and every comparison is explicitly verified to be
fair before any conclusion is drawn.

Shared code lives in the package: [`vnl_experiments/wandb_utils/`](../vnl_experiments/wandb_utils/)
(`fetch`, `comparability`, `style`). Don't duplicate fetch/style logic in a question
folder — add it to `wandb_utils` if it's reusable.

> **⚠️ Known data bug — `body_target_frame` (discovered 2026-07-06).**
> `body_target_frame` is consumed **only by the environment** (`AbsoluteImitation`),
> so the authoritative value is **`config["env_params"]["body_target_frame"]`**.
> The value that also appears under **`config["net_params"]["body_target_frame"]` is
> INERT** — the training scripts (`train_rodent_delays.py`,
> `train_rodent_forward_model.py`) mistakenly set it on `net_config`, where nothing
> reads it; it was only logged. As a result **every AbsoluteImitation run to date was
> trained (and evaluated) with `current_root`**, regardless of the `reference_root`
> shown in `net_params`. When filtering/labelling runs by frame, **always read
> `env_params.body_target_frame`, never `net_params`**. The training scripts were fixed
> on 2026-07-06 (frame now set on `env_config`), so runs after that date are trustworthy
> on either field. This bug invalidates the reference-root vs current-root comparison in
> [`imitation-target-representation/`](imitation-target-representation/) (see its report).
>
> **Related pitfall — the cluster copy drifts from the committed script.** The rule
> "read `env_params`, never the script" applies to *every* env knob, not just the frame.
> The 2026-07-09/07-19 new-XML cohort logged `env_params.body_target_frame =
> current_root` even though the training script at their recorded commits
> (`201d6e11`, `0560d402`) says `reference_root`: the working tree on the cluster had
> been edited (that state was only committed later, as `456fbd7`). WandB stores no
> `diff.patch` for these runs, so `env_params` / the top-level config is the *only*
> record of what actually ran. Same for `torque_actuators` and `walker_xml_path`. See
> [`collision-model-xml/`](collision-model-xml/).

## 1. Layout

One kebab-case folder per question, named after the question:

```
analysis/<question-slug>/
├── extract.py          # fetch from WandB, run comparability check, write data.csv + comparability.txt
├── data.csv            # committed snapshot — one row per included run
├── comparability.txt   # committed programmatic comparability report
├── plot.py             # read data.csv only, write figures/
├── figures/*.png       # committed figures
└── report.md           # question, dataset & comparability, figures, tentative conclusion
```

Start a new question by copying [`_template/`](_template/) and filling it in.

## 2. Two-stage rule (extract vs plot)

- **`extract.py` is the only code that talks to the WandB API** and the only code that
  writes `data.csv`.
- **`plot.py` reads only `data.csv`** — it must **not** import `wandb`.

Why: the committed CSV freezes the exact runs behind a figure (new WandB runs can't
silently change it), and restyling/replotting needs no network and no re-fetch.

## 3. CSV contract

`data.csv` has **one row per included run** (no dedup, no aggregation — that happens in
`plot.py`). Every row must contain, at minimum:

- `wandb_id`, `wandb_name`, `wandb_project` — provenance, so any row traces back to WandB.
- `git_commit` — the code the run was produced with.
- All **experimental variables** for the question (e.g. `delay_k`, `efference_length`).
- All **invariant/comparability fields** (network params, PPO params, env, `total_steps`,
  `actual_step`).
- The **metrics** being plotted (e.g. `episode_reward/mean`).

`vnl_experiments.wandb_utils.fetch.run_record` produces exactly such a flat dict and
always includes the provenance + `git_commit` + `actual_step` columns.

## 4. Comparability protocol (mandatory — programmatic *and* manual)

Before plotting two conditions together, confirm the runs are directly comparable.

**Programmatic** — `extract.py` calls
`comparability_report(df, group_col="condition")` and writes the result to
`comparability.txt`. It checks that these are single-valued (overall and within each
condition): `git_commit`, `env`, network params (`latent_size`, `kl_weight`,
`body_target_frame`, …), PPO params (`n_envs`, `total_steps`), and **trained steps**
(`total_steps` *and* the actual `actual_step`). Anything that varies is flagged
`*** VARIES ***`.

**Manual** — the analyst additionally:
- independently inspects the actual run `config` (do **not** trust tags/notes alone);
- reads each run's `tags` and `notes`;
- if `git_commit` differs across runs, runs `git diff <a> <b>` in `vnl-experiments`
  and confirms the differences are additive / do not touch shared training, env,
  network, or reward code (the additive-only check used for the forward-model vs
  efference comparison is the template). Record the verdict in the report.

Any unavoidable mismatch (a delay value present for only one condition, a benign git
difference, etc.) is written up as a **caveat** in `report.md`.

## 5. report.md template

```markdown
# <Question>

## Question
<one or two sentences>

## Dataset & comparability
- Source: WandB project `<entity/project>`, tags `<…>`.
- N runs per condition, delays covered, etc.
- Comparability: <single git commit / explained difference>; all invariants constant
  (see comparability.txt). Caveats: <…>.

## Figures
![<caption>](figures/<name>.png)

## Tentative conclusion
<short, hedged conclusion>
```

## 6. Styling

Always start `plot.py` with:

```python
from vnl_experiments.wandb_utils.style import apply_style, color_for, marker_for, label_for
apply_style()
```

Use `CONDITION_STYLE` / `color_for(condition)` / `marker_for(condition)` so a condition
keeps the same colour and marker across every figure. Use `add_ms_axis(ax, max_delay)`
for delay plots that need the secondary millisecond axis.

## 7. Starting a new question

```bash
cp -r analysis/_template analysis/<question-slug>
```

Then edit `extract.py` (project, tags, conditions, columns), run it, inspect
`comparability.txt`, edit `plot.py`, and write `report.md`.

## 8. The extra evaluation datasets (`eval_results/`)

Besides the training-time WandB metrics, checkpoints can be **re-evaluated offline** with
[`vnl_experiments/delays/eval_runs.py`](../vnl_experiments/delays/eval_runs.py). For each run
it rolls out the deterministic policy on **three datasets** and writes one JSON per run to
[`eval_results/`](../eval_results/) (keyed by `wandb_id`). Mind which directory you read:
the script's default output is `eval_results/` itself, but the current, most complete
re-evaluation set was collected into **`eval_results/eval_results/`** (with the previous set
kept in `old_eval_results/`), and the newer `extract.py`s point at the nested dir. The three
datasets are:

- **`train`** — the 80% training split (same clips the policy trained on);
- **`old_eval`** — the held-out 20% split (unseen clips, *same* 250-frame / 5 s length);
- **`new_eval`** — a separate set of 32 fresh, longer clips (1500 frames / 30 s).

Each JSON carries, per dataset: `episode_reward`, `lifespan_steps`, per-reason
`termination_rate` (incl. `survived`), per-step `errors`, and network `net_metrics` (e.g.
`fm_pred_mse`), plus hierarchical `param_counts`. These are a **second local data source**:
`extract.py` may join them (by `wandb_id`) with the WandB invariants and the committed
`condition` labels — still the only stage that touches data. Worked examples:
[`train-eval-generalization/`](train-eval-generalization/) and
[`forward-model-new-eval/`](forward-model-new-eval/).

**Gotchas (read before using these):**

- **Two clocks.** `clip_length` is in **mocap frames @ 50 Hz** (250 → 5 s, 1500 → 30 s), but
  the policy runs at the **control rate** `ctrl_dt = 0.01 s` (100 Hz). The eval scans the full
  clip in `ceil(frames / (ctrl_dt·mocap_hz)) + 2` control steps — **502** (train/old) and
  **3002** (new_eval). `lifespan_steps` and delays are in control steps (1 step = 10 ms).
- **Raw reward is comparable *within* a dataset, not *across*.** Cumulative `episode_reward`
  scales with clip length (~6× bigger on new_eval), so only compare conditions on the *same*
  dataset with raw reward/lifetime. For cross-dataset comparisons use **length-fair** metrics:
  `reward_per_step` = reward / lifespan, and the per-second **`hazard_rate`** =
  `(1 − survived) / mean-alive-time` (failure terminations only; end-of-clip truncations are
  censored, not events — verified `survived + failure-reasons == 1`). Prefer `hazard_rate` over
  a raw survival fraction across datasets: survival fraction penalises longer clips for simply
  having more chances to fail, whereas the hazard is clip-length-invariant.
- **`new_eval` is noisy** — only 32 clips, single seed per cell. Read its curves for trend, not
  point values.
- **The eval is not bit-reproducible.** MuJoCo Warp's GPU physics is nondeterministic, and over
  a 502-step rollout that amplifies: re-evaluating the *same* checkpoint with the *same* seed
  moves `episode_reward` by ~1% and can flip individual clips between surviving and
  terminating. Per-clip quantities are worst hit — every `std`, and `termination_rate` (at 32
  clips, one clip flipping is 3 percentage points). So don't read a small difference between
  two eval passes as signal, and don't expect a re-run to diff clean against a committed JSON.
- The same comparability protocol (§4) still applies; additionally sanity-check that the
  restored `checkpoint_step` and `clip_length`/rollout horizons came out as expected (the eval
  script and the example `extract.py`s print these).

### End-of-training eval (runs from 2026-08-10 onward)

Training runs now evaluate themselves when they finish: `train_rodent_delays.py` and
`train_rodent_forward_model.py` run **the same evaluation described above** on the
just-trained (in-memory) network, through the shared
[`vnl_experiments/delays/evaluation.py`](../vnl_experiments/delays/evaluation.py). The record
has an identical schema to the batch output. Two consequences for analyses:

- **The record is written to `{ckpt_dir}/eval.json`**, so it rsyncs back from the cluster
  along with the checkpoint. Gather these into the flat, `wandb_id`-keyed directory the
  extract scripts glob over with:

  ```bash
  ../.venv/bin/python -m vnl_experiments.delays.eval_runs --collect
  ```

  (target defaults to `eval_results/eval_results/`; `--override` replaces existing files.)

- **The headline numbers are also pushed to the WandB run summary** under `final_eval/…` —
  e.g. `final_eval/old_eval/episode_reward/mean`,
  `final_eval/new_eval/termination_rate/survived`,
  `final_eval/old_eval/net_metrics/3/action/1/fm_pred_mse`, `final_eval/params/total`. For
  new runs an `extract.py` can read these straight from the run summary via `wandb_utils`,
  with no local JSON needed. Runs from before this date have no `final_eval/*` keys — fall
  back to `eval_results/`.

The inline eval measures the **in-memory** network at `total_steps`. Training deliberately
does *not* write an extra checkpoint when it finishes, so strictly the evaluated weights are
whatever the last on-grid checkpoint would have been plus any steps since. With
`total_steps = 600M` and `checkpoint_every_steps = 50M` the two coincide exactly; when they
don't, the training script warns and names the gap, and the inline record can then differ
slightly from what a later `eval_runs.py` pass produces (which restores the newest checkpoint
on disk). The gap is a fraction of a percent of training — smaller than the run-to-run
nondeterminism above.

`eval_runs.py` stays the authority for **cross-run** comparisons: it is the only way to
re-evaluate the whole cohort under one version of the eval code (`--override`). An inline
`eval.json` is frozen at whatever the eval code looked like when that run finished, so when a
figure compares runs trained at different times, prefer a single batch re-evaluation over a
mix of inline records. Note also that crashed, preempted and resumed runs never reach the
inline eval, so `eval_runs.py` remains the way to fill those gaps.
