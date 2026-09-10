# <Question>

## Question

<One or two sentences. State the question as a question, and say what would count as an
answer.>

## Dataset & comparability

- **Source:** WandB `<entity/project>`, tags `<…>`, selected by the `CONDITIONS` in
  `extract.py` and frozen in `runs.csv`.
- **Conditions:**

| condition | task | n | seeds | delays | network | commits |
|---|---|---|---|---|---|---|
| `<name>` | | | | | | |

- **Reward reported:** <which reward every number below is, and *which side of the
  2026-09-10 eval-wrapper fix the runs are on*. Before it, `eval/*` is scaled x10 and the
  episodes are truncated by the training EpisodeWrapper's random phase; after it, it is
  unscaled and full-length. Runs either side are not poolable. Training-side reward
  carries `reward_scale = 10` on both sides. If any y-axis is a ratio or a fraction
  rather than raw reward, say in one line why that was unavoidable -- across-task
  comparison is not on its own a good enough reason, since small multiples do that job.>
- **Tasks:** <which of the nine, and whether reward is being compared across them. It is
  not the same quantity in two tasks.>
- **Artifacts used:** `REQUIRES = [...]`; see `coverage.txt` for per-condition coverage.
  <State explicitly if any condition is short of full coverage, and what that means for
  the conclusions.>
- **Programmatic comparability:** see `comparability.txt`. <Which invariants hold; which
  are flagged and why that is acceptable.>
- **Manual comparability:** <the analyst's own check — configs inspected directly, tags
  and notes read, and if `git_commit` varies, the `git diff <a> <b>` verdict on whether
  the differences touch shared training/env/network/reward code.>
- **Caveats:** <confounds, single-seed cells, unmatched delays, GPU differences, …>

## Figures

![<caption>](figures/<name>.png)

<One or two sentences per figure saying what to look at, not just what is plotted.>

## Tentative conclusion

<Short and hedged. Separate what the data shows from what it suggests.>

## Follow-ups

- <What would sharpen or falsify this.>

---

*Reproduce:* `../.venv/bin/python analysis/dm_control_suite/<question-slug>/extract.py && ../.venv/bin/python analysis/dm_control_suite/<question-slug>/plot.py`
(add `--sync --refresh` to the extract to pull in runs added since `runs.csv` was frozen).
