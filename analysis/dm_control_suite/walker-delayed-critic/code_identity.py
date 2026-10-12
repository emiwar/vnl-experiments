"""Was every run in this cohort built and trained by the same code, apart from the critic?

The cohort spans seven vnl-experiments commits and three nnx-ppo commits, and the
delayed-critic arm is *entirely* on the newest of each (``a37ce64`` / ``21d6643``) while the
privileged arm is entirely on older ones. So "the arms differ only in the critic" is not
free: commit is perfectly confounded with the manipulation, and has to be ruled out as the
explanation. Pattern: ``../../rodent/efference-copy-vs-proprioception/code_identity.py``
(README §6) -- hash the load-bearing code at each commit, docstrings stripped.

Two subjects *did* change at ``a37ce64``, and changed on purpose: ``a37ce64`` is the commit
that added the non-privileged critic. For those, "identical" means *the privileged code
path is untouched*: the hash is taken after **neutralising** the addition --

* ``make_delayed_mlp_actor_critic`` -- drop the ``privileged_critic`` parameter and the
  ``if not privileged_critic:`` block (which returns early, so the rest of the body is
  exactly what a privileged build runs);
* ``build_flat_delay_network`` -- drop the ``privileged_critic=`` keyword it passes;
* ``EfferenceCopy.__call__`` -- replace ``if isinstance(..., PPONetworkOutput): ... else:
  <old>`` by ``<old>``, and drop the import. Only a ``PPOAdapter`` inner takes the new
  branch, and in the privileged layout the inner is a ``Sequential`` ending in a sampler.

The neutralised hash is printed beside the plain one, so the change itself stays visible.

Out of scope, and covered elsewhere:

* the env stack (``registry._dmc_builder``, ``actuator_mode``, ``nan_guard``) changed across
  these commits too. ``../walker-forward-model-1b/nan_guard_inert.txt`` shows the NaN guard
  never fired on this task and stack, and ``../walker-joint-stiffness/`` that ``servo_kp =
  0`` leaves the model unpatched. The *empirical* check that nothing in the stack moved is
  the delay-0 control in report.md: at ``delay_k = 0`` the two critics see identical inputs.
* ``ppo_step`` / ``train_ppo`` gained NaN diagnostics at ``5314279`` -- a non-finite count
  computed alongside the update and a host-side check after it. Hashed here and expected to
  differ there; report.md quotes the diff.
* ``repos.vnl_experiments.dirty`` is True for many runs, including all 19 delayed-critic
  ones, so the commit does not fully identify what ran (README §4). Bounded empirically by
  the same delay-0 control.

    ../.venv/bin/python analysis/dm_control_suite/walker-delayed-critic/code_identity.py

Writes ``code_identity.txt``. Reads git and ``data.csv`` only.
"""

import ast
import hashlib
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
VNL = HERE.parents[2]
NNX = VNL.parent / "nnx-ppo"
OUT = HERE / "code_identity.txt"
ABSENT = "absent"

#: (repo, path, top-level def or class name or None for the whole file, neutralise?)
SUBJECTS = [
    ("nnx", "nnx_ppo/algorithms/ppo.py", "ppo_loss", False),
    ("nnx", "nnx_ppo/algorithms/ppo.py", "gae", False),
    ("nnx", "nnx_ppo/algorithms/ppo.py", "ppo_step", False),
    ("nnx", "nnx_ppo/algorithms/rollout.py", "unroll_env", False),
    ("nnx", "nnx_ppo/algorithms/rollout.py", "single_transition", False),
    ("nnx", "nnx_ppo/algorithms/rollout.py", "eval_rollout", False),
    ("nnx", "nnx_ppo/networks/normalizer.py", None, False),
    ("nnx", "nnx_ppo/networks/sampling_layers.py", None, False),
    ("nnx", "nnx_ppo/networks/delay.py", None, False),
    ("nnx", "nnx_ppo/networks/adapter.py", None, False),
    ("nnx", "nnx_ppo/networks/containers.py", None, False),
    ("nnx", "nnx_ppo/networks/factories.py", None, False),
    ("vnl", "vnl_experiments/delays/make_delayed_networks.py",
     "make_delayed_mlp_actor_critic", True),
    ("vnl", "vnl_experiments/delays/network_builders.py", "build_flat_delay_network", True),
    ("vnl", "vnl_experiments/delays/network_builders.py", "_parse_net_params", False),
    ("vnl", "vnl_experiments/delays/efference_copy.py", None, True),
]


def show(repo: Path, commit: str, path: str) -> str | None:
    out = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"],
                         capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            b = node.body
            if (b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant)
                    and isinstance(b[0].value.value, str)):
                node.body = b[1:] or [ast.Pass()]
    return tree


class Neutralise(ast.NodeTransformer):
    """Remove the a37ce64 additions so the privileged path can be compared."""

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        kept = [(a, d) for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                if a.arg != "privileged_critic"]
        node.args.kwonlyargs = [a for a, _ in kept]
        node.args.kw_defaults = [d for _, d in kept]
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        src = ast.unparse(node.test)
        if src == "not privileged_critic":
            return None
        if "PPONetworkOutput" in src:
            return node.orelse
        return node

    def visit_keyword(self, node):
        return None if node.arg == "privileged_critic" else node

    def visit_alias(self, node):
        return None if node.name == "PPONetworkOutput" else node

    def visit_ImportFrom(self, node):
        self.generic_visit(node)
        return node if node.names else None


def extract(text: str | None, name: str | None) -> ast.AST | None:
    if text is None:
        return None
    tree = ast.parse(text)
    if name is None:
        return tree
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.Module(body=[node], type_ignores=[])
    return None


def digest(tree: ast.AST | None) -> str:
    if tree is None:
        return ABSENT
    return hashlib.sha256(ast.unparse(strip_docstrings(tree)).encode()).hexdigest()[:10]


def main() -> None:
    data = pd.read_csv(HERE / "data.csv")
    commits = {
        "vnl": sorted(data["git_commit"].dropna().unique(),
                      key=lambda c: subprocess.run(
                          ["git", "-C", str(VNL), "log", "-1", "--format=%ct", c],
                          capture_output=True, text=True).stdout),
        "nnx": sorted(data["nnx_ppo_commit"].dropna().unique(),
                      key=lambda c: subprocess.run(
                          ["git", "-C", str(NNX), "log", "-1", "--format=%ct", c],
                          capture_output=True, text=True).stdout),
    }
    repos = {"vnl": VNL, "nnx": NNX}
    count = {"vnl": data["git_commit"].value_counts(),
             "nnx": data["nnx_ppo_commit"].value_counts()}
    arms = {"vnl": data.groupby("git_commit")["critic"].agg(lambda s: "/".join(sorted(set(s)))),
            "nnx": data.groupby("nnx_ppo_commit")["critic"].agg(
                lambda s: "/".join(sorted(set(s))))}

    lines = ["Is the load-bearing code identical across the cohort's commits?", "",
             "sha256[:10] of each subject with docstrings stripped and re-unparsed. "
             "'neutralised' rows",
             "first remove the a37ce64 non-privileged-critic additions (see the "
             "docstring).", ""]
    for repo in ("vnl", "nnx"):
        lines.append(f"{repo} commits (oldest first), runs, critic arm:")
        for c in commits[repo]:
            lines.append(f"  {c[:7]}  n={int(count[repo].get(c, 0)):2d}  {arms[repo][c]}")
        lines.append("")

    verdicts = []
    for repo in ("vnl", "nnx"):
        cs = commits[repo]
        lines.append(f"-- {repo} " + "-" * 60)
        lines.append(f"{'subject':<56s} " + " ".join(f"{c[:7]:>10s}" for c in cs)
                     + "   verdict")
        for r, path, name, neutral in SUBJECTS:
            if r != repo:
                continue
            label = f"{path.split('/')[-1]}:{name or '*'}"
            variants = [("plain", False)] + ([("neutralised", True)] if neutral else [])
            for tag, do in variants:
                row = []
                for c in cs:
                    tree = extract(show(repos[repo], c, path), name)
                    if do and tree is not None:
                        tree = ast.fix_missing_locations(Neutralise().visit(tree))
                    row.append(digest(tree))
                present = {h for h in row if h != ABSENT}
                verdict = ("identical" if len(present) == 1
                           else f"{len(present)} versions")
                lab = label + ("" if tag == "plain" else "  [neutralised]")
                lines.append(f"{lab:<56s} " + " ".join(f"{h:>10s}" for h in row)
                             + f"   {verdict}")
                verdicts.append((repo, lab, neutral, tag, verdict))
        lines.append("")

    must_match = [v for v in verdicts
                  if (v[2] and v[3] == "neutralised") or (not v[2] and v[1] != "ppo.py:ppo_step")]
    bad = [v[1] for v in must_match if v[4] != "identical"]
    lines.append("Verdict")
    if bad:
        lines.append("  *** NOT IDENTICAL: " + ", ".join(bad) + " ***")
    else:
        lines.append("  Every subject that defines the networks, the sampler, the "
                     "normaliser, the rollout,")
        lines.append("  GAE and the PPO loss is identical at every commit -- for the a37ce64 "
                     "additions, after")
        lines.append("  removing them, i.e. the privileged build path is byte-for-byte the "
                     "old one. ppo_step is")
        lines.append("  expected to differ (NaN diagnostics, 5314279); see report.md.")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
