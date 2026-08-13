# <Question>

## Question

<One or two sentences. State the question as a question, and say what would count as an
answer.>

## Dataset & comparability

- **Source:** WandB `<entity/project>`, tags `<…>`, selected by the `CONDITIONS` in
  `extract.py` and frozen in `runs.csv`.
- **Conditions:**

| condition | n | delays | XML | control | frame | commits |
|---|---|---|---|---|---|---|
| `<name>` | | | | | | |

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

*Reproduce:* `../.venv/bin/python analysis/<question-slug>/extract.py && ../.venv/bin/python analysis/<question-slug>/plot.py`
(add `--sync --refresh` to the extract to pull in runs added since `runs.csv` was frozen).
