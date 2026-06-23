# Analysis pipeline

This folder holds **reproducible WandB analyses**. Each scientific question gets its
own folder. The goal: a figure is reproducible from a committed CSV snapshot even
after more runs are added to WandB, and every comparison is explicitly verified to be
fair before any conclusion is drawn.

Shared code lives in the package: [`vnl_experiments/wandb_utils/`](../vnl_experiments/wandb_utils/)
(`fetch`, `comparability`, `style`). Don't duplicate fetch/style logic in a question
folder — add it to `wandb_utils` if it's reusable.

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
