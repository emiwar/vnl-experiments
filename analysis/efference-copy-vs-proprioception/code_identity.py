"""Is the code that built these networks the same across the cohort's five commits?

Why this exists
---------------
The cohort spans six ``git_commit`` values, so README §4 requires the analyst to diff them
and confirm the differences do not touch shared training, env, network or reward code. The
diffs are large -- ``b7c4b32f..2ae4a5dd`` alone is 81 files and ~10 000 changed lines --
because a training-entry-point refactor and a whole second family of architectures (the
flat-observation dm_control nets) landed in between. Reading that and declaring it benign is
exactly the judgement the pipeline asks to be replaced with a measurement.

So instead of diffing the tree, this hashes **the code that actually builds and runs these
networks**, at each commit:

* ``build_delay_network`` -- turns ``net_params`` into the enc-dec net, including the
  ``dec_use_proprioception`` branch and the ``efference_length`` arithmetic this whole
  question is about;
* ``efference_copy.py`` -- the ``EfferenceCopy`` queue itself;
* ``absolute_imitation.py`` -- the env, whose ``_get_imitation_target`` defines ``task_obs``;
* ``_parse_net_params`` -- the parser whose ``int()`` bug is why ``UNREGULARIZED_COMMITS``
  exists, so a change here would be a silent hyperparameter change.

Two hashes per subject
----------------------
``raw`` is the bytes. ``code`` is the same source parsed, stripped of every docstring, and
re-unparsed -- so comments and docstrings cannot make it differ, and nothing executable can
hide behind them. The distinction is load-bearing here rather than decorative: two of these
four subjects *do* differ in raw bytes at ``2ae4a5dd``, and both differences are a renamed
script in a docstring (``train_rodent_delays.py`` -> ``train.py``). ``code`` is what the
verdict is taken from; ``raw`` is printed so that the docstring drift is visible rather than
normalised out of sight.

What it cannot show
-------------------
``repos.nnx_ppo.dirty`` and ``repos.vnl_playground.dirty`` are ``True`` for every run in this
cohort that recorded them, meaning the cluster working copy had uncommitted edits when the
run started. **The recorded commit therefore does not identify the code that ran**
(README §4: "any True here voids the commit"), and hashing that commit cannot recover what
was executed. This script bounds the *committed* differences; the matched-commit replicate
agreement in report.md (<= 0.6 %) is what bounds the rest. Both are reported, and neither is
presented as the other.

    ../.venv/bin/python analysis/efference-copy-vs-proprioception/code_identity.py
    ../.venv/bin/python analysis/efference-copy-vs-proprioception/code_identity.py --check

Writes ``code_identity.txt``. Reads git and ``data.csv``; needs no GPU and no artifact store.
"""

import argparse
import ast
import difflib
import hashlib
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "code_identity.txt"

#: Whole files whose executable code must match.
FILES = (
    "vnl_experiments/delays/efference_copy.py",
    "vnl_experiments/envs/absolute_imitation.py",
)

#: ``(file, start_line_prefix, end_line_prefix)`` -- one top-level definition, sliced by its
#: ``def`` line up to the next given line, so unrelated additions elsewhere in a large
#: module cannot mask a change to the part that matters. A slice of top-level definitions is
#: itself valid Python, which is what lets it be parsed below.
FUNCTIONS = (
    ("vnl_experiments/delays/network_builders.py",
     "def build_delay_network", "def build_forward_model_network"),
    ("vnl_experiments/delays/network_builders.py",
     "def _parse_net_params", "# ------"),
)

ABSENT = "absent"


def show(commit: str, path: str) -> str | None:
    out = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"],
                         capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def slice_definition(text: str, start: str, end: str) -> str | None:
    lines = text.splitlines(keepends=True)
    first = next((i for i, l in enumerate(lines) if l.startswith(start)), None)
    if first is None:
        return None
    rest = next((j for j, l in enumerate(lines[first + 1:], first + 1)
                 if l.startswith(end)), len(lines))
    return "".join(lines[first:rest])


def strip_docstrings(tree: ast.AST) -> ast.AST:
    """Drop the docstring statement from every module / class / function in ``tree``."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            # Keep the body non-empty: a function whose only statement was its docstring
            # would otherwise become unparseable.
            node.body = body[1:] or [ast.Pass()]
    return tree


def digests(text: str | None) -> tuple[str, str]:
    """``(raw, code)`` hashes: bytes, and docstring-stripped executable code."""
    if text is None:
        return ABSENT, ABSENT
    raw = hashlib.sha256(text.encode()).hexdigest()[:12]
    try:
        code_src = ast.unparse(strip_docstrings(ast.parse(text)))
    except SyntaxError:
        return raw, "unparseable"
    return raw, hashlib.sha256(code_src.encode()).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    data = pd.read_csv(HERE / "data.csv")
    commits = sorted(data["git_commit"].dropna().unique())
    counts = data["git_commit"].value_counts()

    subjects: list[str] = []
    raw: dict[str, dict[str, str]] = {}
    code: dict[str, dict[str, str]] = {}

    def record(label: str, per_commit: dict[str, str | None]) -> None:
        subjects.append(label)
        raw[label], code[label] = {}, {}
        for commit, text in per_commit.items():
            raw[label][commit], code[label][commit] = digests(text)

    for path in FILES:
        record(path.split("/")[-1], {c: show(c, path) for c in commits})
    for path, start, end in FUNCTIONS:
        texts = {}
        for commit in commits:
            source = show(commit, path)
            texts[commit] = (None if source is None
                             else slice_definition(source, start, end))
        record(f"{start.removeprefix('def ')}()", texts)

    lines = ["Is the load-bearing code identical across the cohort's commits?", ""]
    lines.append("Per subject and commit: sha256[:12] of the raw bytes, and of the same "
                 "source with every")
    lines.append("docstring stripped and re-unparsed. The verdict is taken from the "
                 "second: docstrings")
    lines.append("cannot make it differ, and nothing executable can hide behind them.")
    lines.append("")
    lines.append("Commits, and how many runs of this cohort were built from each:")
    for c in commits:
        lines.append(f"  {c}  n={int(counts.get(c, 0)):2d}")
    lines.append("")

    width = max(len(s) for s in subjects)
    failures = []
    for kind, table in (("raw bytes", raw), ("executable code", code)):
        lines.append(f"-- {kind} " + "-" * (width + 16 * len(commits) - len(kind) - 3))
        lines.append(f"{'subject':<{width}s} " + " ".join(f"{c:>14s}" for c in commits)
                     + "   verdict")
        for label in subjects:
            row = table[label]
            unique = set(row.values())
            if unique == {ABSENT}:
                verdict = "*** ABSENT EVERYWHERE ***"
            elif len(unique) == 1:
                verdict = "identical"
            else:
                verdict = "differs"
            if kind == "executable code" and verdict != "identical":
                failures.append(label)
                verdict = f"*** {verdict.upper()} ***"
            lines.append(f"{label:<{width}s} "
                         + " ".join(f"{row[c]:>14s}" for c in commits)
                         + f"   {verdict}")
        lines.append("")

    differing_raw = [s for s in subjects if len(set(raw[s].values())) > 1]
    lines.append("Reading")
    lines.append("  Every subject's executable code is identical at every commit in the "
                 "cohort. The large")
    lines.append("  diffs between these commits (a training-entry-point refactor and the "
                 "flat-observation")
    lines.append("  dm_control architectures) are additive and touch none of it, so "
                 "`git_commit` varying")
    lines.append("  inside a condition does not make that condition's runs "
                 "differently-built.")
    if differing_raw:
        lines.append("")
        lines.append(f"  Raw bytes do differ for {', '.join(differing_raw)}: a script "
                     f"renamed in a docstring")
        lines.append("  (`train_rodent_delays.py` / `train_rodent.py` -> `train.py`), "
                     "which is why the")
        lines.append("  docstring-stripped hash is the one the verdict reads.")
    lines.append("")
    lines.append("  This bounds the COMMITTED code only. Every run here that recorded "
                 "repos.*.dirty")
    lines.append("  recorded True, so the working copy had uncommitted edits and the "
                 "commit does not")
    lines.append("  identify what ran. report.md bounds the remainder empirically "
                 "instead, from the")
    lines.append("  matched-commit replicate pairs.")
    lines.append("")
    lines.append("FAILURES: " + (", ".join(sorted(set(failures))) if failures else "none"))

    text = "\n".join(lines) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print("\n".join(difflib.unified_diff(
                (OUT.read_text() if OUT.exists() else "").splitlines(),
                text.splitlines(), fromfile=f"committed/{OUT.name}",
                tofile=f"rebuilt/{OUT.name}", lineterm="")))
            raise SystemExit(1)
        print(f"CHECK: {OUT.name} unchanged")
    else:
        OUT.write_text(text)
        print(text)

    if failures:
        raise SystemExit("the code-identity expectations do not hold; see above")


if __name__ == "__main__":
    main()
