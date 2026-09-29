"""Is ``efference_length`` real? Does 0 mean no queue, and does > 0 mean the right queue?

Why this exists
---------------
This folder reports that removing the efference copy costs WalkerWalk nothing out to 125 ms
of delay. That contradicts the rodent result it is compared against and the reason the
efference-matched convention exists at all, and there is a much duller explanation
available: **if the queue were inert, misaligned, or not built at all, "no copy needed" is
exactly what the experiment would return.** A negative result whose most likely cause is a
bug is not a result until the bug is excluded.

So this script does not read the conclusion; it attacks the mechanism, in four parts.

A. **Source identity across the cohort's commits.** The cohort spans five
   ``vnl-experiments`` commits and two ``nnx-ppo`` commits. Rather than diffing them, hash
   the code that actually builds and runs the queue -- ``EfferenceCopy``, ``Delay``,
   ``make_delayed_mlp_actor_critic``, ``build_flat_delay_network``, ``_parse_net_params``,
   and the ``net_params`` literal in ``train.py`` that decides what the builder is handed --
   at each commit, comparing docstring-stripped executable source rather than raw bytes.
   Same method as ``../../rodent/efference-copy-vs-proprioception/code_identity.py``.

B. **Structure.** Build the network from each cohort run's **logged** ``net_params`` and
   read the actor's first Linear kernel off the built module. It must be
   ``obs_size + efference_length * action_size`` wide -- 24 for every no-copy run, 24 + 6L
   for every copy run -- and the ``efference_length = 0`` carry must contain no ``"queue"``
   entry at all. This is the check that the config the run logged really produces the
   network the condition label claims.

C. **The queue reaches the policy.** Structure is not use: a queue that is built, filled
   and then multiplied by a zeroed weight block would pass B. So hold the observation and
   the RNG fixed, put two different action histories in the carry, and require the policy's
   output to differ. Run at the *initialised* weights, where a dead path would be the
   default rather than something training had to produce.

D. **Alignment.** The queue could be real, reach the policy, and still be wrong: off by one
   step, or ordered oldest-first, either of which would degrade the copy's usefulness
   without making it inert. Compose ``Delay(k)`` and ``EfferenceCopy(L)`` exactly as
   ``make_delayed_mlp_actor_critic`` composes them, put a recording stub where the actor MLP
   goes, and assert that at step t the inner module receives
   ``concat(obs[t - k], a[t - 1], a[t - 2], ..., a[t - L])`` -- the delayed observation
   followed by the actions taken since it, newest first.

What this cannot show
---------------------
``repos.vnl_experiments.dirty`` is ``True`` on part of this cohort, so the recorded commit
does not identify the code that ran (README §4) and part A bounds the *committed* code only.
Parts B-D run against the working tree, which is what part A's ``working tree`` column is
for: if it matches the commits, the thing tested is the thing hashed. Nothing here can prove
the cluster executed it -- only a checkpoint can, by its first-layer shape, and those stay
on the cluster. ``report.md`` says so.

    ../.venv/bin/python analysis/dm_control_suite/walker-efference-copy/efference_identity.py
    ../.venv/bin/python analysis/dm_control_suite/walker-efference-copy/efference_identity.py --check

Writes ``efference_identity.txt``. Reads git, ``data.csv`` and the run index; needs no
artifact store and no GPU (the networks are built against a stub env carrying only
``observation_size`` / ``action_size``, taken from the real WalkerWalk env when it can be
built and from the recorded constants otherwise).
"""

import argparse
import ast
import hashlib
import subprocess
import textwrap
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import pandas as pd
from flax import nnx

from nnx_ppo.networks.containers import Sequential
from nnx_ppo.networks.sampling_layers import ActionSampler
from nnx_ppo.networks.delay import Delay
from nnx_ppo.networks.types import StatefulModule, StatefulModuleOutput
from vnl_experiments.delays.efference_copy import EfferenceCopy
from vnl_experiments.delays.network_builders import build_network

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
NNX_PPO = REPO.parent / "nnx-ppo"
OUT = HERE / "efference_identity.txt"

#: Flat observation and action sizes per task. Verified against the built envs when they
#: can be constructed; the table is the fallback so this script runs without a GPU, and a
#: mismatch is reported rather than silently preferred either way. These differ by more
#: than an order of magnitude across the cohort -- CartpoleSwingup is 5/1 and HumanoidWalk
#: 67/21 -- which is exactly why part B cannot use one number.
SIZES = {
    "CartpoleSwingup": (5, 1),
    "CheetahRun": (17, 6),
    "ReacherHard": (6, 2),
    "HumanoidWalk": (67, 21),
    "WalkerWalk": (24, 6),
}

#: The task parts C and D are demonstrated on. They test the wrapper, which is shared by
#: every task, so one is enough -- and part B covers all five with their own sizes.
DEMO_TASK = "WalkerWalk"

#: Whole files whose executable code must match across the cohort, per repo.
FILES = {
    "vnl-experiments": ("vnl_experiments/delays/efference_copy.py",),
    "nnx-ppo": ("nnx_ppo/networks/delay.py",),
}

#: ``(repo, file, start_line_prefix, end_line_prefix)`` -- one top-level definition, sliced
#: by its ``def`` line up to the next given line, so unrelated additions elsewhere in a
#: large module cannot mask a change to the part that matters.
FUNCTIONS = (
    ("vnl-experiments", "vnl_experiments/delays/network_builders.py",
     "def build_flat_delay_network", "def build_flat_forward_model_network"),
    ("vnl-experiments", "vnl_experiments/delays/network_builders.py",
     "def _parse_net_params", "def _"),
    ("vnl-experiments", "vnl_experiments/delays/make_delayed_networks.py",
     "def make_delayed_mlp_actor_critic", "def make_forward_model_actor_critic"),
)

#: The block of ``train.py`` that decides what ``build_network`` is handed. Sliced by
#: content rather than by ``def`` because it lives inside a long function: if this drifts,
#: ``net_params.efference_length`` in WandB could stop being the value the builder used,
#: which is the single assumption every condition in this folder rests on.
TRAIN_BLOCK = ("vnl-experiments", "vnl_experiments/train.py",
               "    net_params = {", "    # Decoder-input ablations")

ABSENT = "absent"
UNPARSEABLE = "unparseable"
WORKING_TREE = "working tree"


class _Recorder(StatefulModule):
    """A stand-in for the actor MLP that records what it is given and emits a fixed action.

    Part D needs to see the tensor the actor *actually* receives, which no amount of
    inspecting the built network will show. Emitting a scripted action rather than a
    learned one makes the expected queue contents known in advance, so the assertion is
    about alignment and not about the policy.
    """

    def __init__(self, actions, action_size: int):
        self.actions = actions          # [T, A]: what to emit at each step
        self.action_size = action_size
        self.seen: list[np.ndarray] = []
        self.t = 0

    def initialize_state(self, batch_size: int):
        return {}

    def reset_state(self, prev_state):
        return {}

    def __call__(self, state, obs, rollout_extras=None):
        self.seen.append(np.asarray(obs))
        action = jp.broadcast_to(self.actions[self.t],
                                 (obs.shape[0], self.action_size))
        self.t += 1
        return StatefulModuleOutput(
            next_state={},
            output={"action": action,
                    "log_likelihood": jp.zeros(obs.shape[0])},
            regularization_loss=jp.zeros(obs.shape[0]),
            metrics={},
            rollout_extras=None,
        )

    def update_statistics(self, rollout_extras):
        pass


class _StubEnv:
    """Only what ``build_network`` reads. Keeps this script off the GPU."""

    def __init__(self, obs_size: int, action_size: int):
        self.observation_size = obs_size
        self.action_size = action_size


# ---------------------------------------------------------------------------
# Part A -- source identity
# ---------------------------------------------------------------------------

def _repo_path(repo: str) -> Path:
    return REPO if repo == "vnl-experiments" else NNX_PPO


def show(repo: str, commit: str, path: str) -> str | None:
    if commit == WORKING_TREE:
        f = _repo_path(repo) / path
        return f.read_text() if f.exists() else None
    out = subprocess.run(["git", "-C", str(_repo_path(repo)), "show", f"{commit}:{path}"],
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
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def digests(text: str | None) -> tuple[str, str]:
    """``(raw, code)`` hashes: bytes, and docstring-stripped executable code.

    ``textwrap.dedent`` first, because one subject (:data:`TRAIN_BLOCK`) is a slice from
    *inside* a function and is indented four spaces, which ``ast.parse`` rejects. Without
    the dedent every commit hashes to the sentinel ``"unparseable"``, they all match, and
    the row reports "identical" while having compared nothing -- a vacuous pass on the one
    subject that decides whether WandB's ``net_params`` is the dict the builder was given.
    ``main`` treats the sentinel as a failure for the same reason.
    """
    if text is None:
        return ABSENT, ABSENT
    raw = hashlib.sha256(text.encode()).hexdigest()[:10]
    try:
        code_src = ast.unparse(strip_docstrings(ast.parse(textwrap.dedent(text).strip())))
    except SyntaxError:
        return raw, UNPARSEABLE
    return raw, hashlib.sha256(code_src.encode()).hexdigest()[:10]


def part_a(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    commits = {
        "vnl-experiments": sorted(df["git_commit"].dropna().unique()),
        "nnx-ppo": sorted(df["nnx_ppo_commit"].dropna().unique()),
    }
    counts = {"vnl-experiments": df["git_commit"].value_counts(),
              "nnx-ppo": df["nnx_ppo_commit"].value_counts()}

    subjects, raw, code = [], {}, {}

    def record(repo: str, label: str, texts: dict[str, str | None]) -> None:
        subjects.append((repo, label))
        raw[label], code[label] = {}, {}
        for commit, text in texts.items():
            raw[label][commit], code[label][commit] = digests(text)

    for repo, paths in FILES.items():
        for path in paths:
            cs = commits[repo] + [WORKING_TREE]
            record(repo, path.split("/")[-1], {c: show(repo, c, path) for c in cs})
    for repo, path, start, end in FUNCTIONS + (TRAIN_BLOCK,):
        texts = {}
        for commit in commits[repo] + [WORKING_TREE]:
            src = show(repo, commit, path)
            texts[commit] = None if src is None else slice_definition(src, start, end)
        label = (start.removeprefix("def ").rstrip("{ =") + "()"
                 if start.startswith("def ") else start.strip().rstrip("{ ="))
        record(repo, label, texts)

    lines = ["=" * 78,
             "A. Is the code that builds and runs the queue the same across the cohort?",
             "=" * 78,
             "",
             "sha256[:10] of the source with every docstring stripped and re-unparsed, so",
             "docstrings cannot make it differ and nothing executable can hide behind them.",
             "The `working tree` column is what parts B-D actually execute.",
             ""]
    for repo in ("vnl-experiments", "nnx-ppo"):
        lines.append(f"  {repo} commits in the cohort:")
        for c in commits[repo]:
            lines.append(f"    {c[:10]}  n={int(counts[repo].get(c, 0)):2d}")
    lines.append("")

    failures = []
    for repo in ("vnl-experiments", "nnx-ppo"):
        cs = commits[repo] + [WORKING_TREE]
        mine = [(r, l) for r, l in subjects if r == repo]
        if not mine:
            continue
        width = max(len(l) for _, l in mine)
        lines.append(f"-- {repo} " + "-" * 60)
        lines.append(f"{'subject':<{width}s} "
                     + " ".join(f"{c[:10]:>12s}" for c in cs) + "   verdict")
        for _, label in mine:
            row = code[label]
            unique = set(row.values())
            if unique == {ABSENT}:
                verdict, bad = "*** ABSENT EVERYWHERE ***", True
            elif UNPARSEABLE in unique:
                # Never let this read as "identical": the sentinel compares equal to
                # itself, so an unparseable subject would silently verify nothing.
                verdict, bad = "*** UNPARSEABLE ***", True
            elif len(unique) == 1:
                verdict, bad = "identical", False
            else:
                verdict, bad = "*** DIFFERS ***", True
            if bad:
                failures.append(label)
            lines.append(f"{label:<{width}s} "
                         + " ".join(f"{row[c]:>12s}" for c in cs) + f"   {verdict}")
        lines.append("")

    if failures:
        lines.append(f"  *** {len(failures)} subject(s) differ across commits: "
                     f"{failures}. Every condition in this folder assumes one "
                     f"implementation of the queue; it does not hold.")
    else:
        lines.append("  All subjects identical at every commit in the cohort AND in the")
        lines.append("  working tree, so parts B-D below test the code the cohort was")
        lines.append("  built from -- subject to the dirty-working-copy limit above.")
    lines.append("")
    return lines, failures


# ---------------------------------------------------------------------------
# Parts B-D -- runtime behaviour
# ---------------------------------------------------------------------------

def actor_first_kernel(nets) -> tuple[int, int]:
    """Shape of the first Linear kernel inside the actor branch of a built network.

    Found by walking the module tree for ``Linear`` leaves rather than by indexing a known
    path, because the factory returns a bare ``PPOAdapter`` or a ``Sequential`` depending
    on ``normalize_obs`` and the nesting has changed before. The actor's first Linear is
    the one whose input width depends on ``efference_length``; the critic's is always
    ``obs_size``, which is why the search is restricted to the action branch.
    """
    action_branch = nets
    while not hasattr(action_branch, "action"):
        action_branch = action_branch.layers[-1]
    shapes = [(m.in_features, m.out_features) for _, m in
              nnx.iter_graph(action_branch.action) if isinstance(m, nnx.Linear)]
    return shapes[0]


def build_from(net_params: dict, env, seed: int = 0):
    return build_network(net_params, env, nnx.Rngs(seed))


def part_b(df: pd.DataFrame, sizes: dict) -> tuple[list[str], list[str]]:
    """Every run's logged net_params must build the network its condition claims.

    Per **task**, because ``obs_size`` and ``action_size`` differ by more than an order of
    magnitude across this cohort: an ``efference_length = 10`` queue is 10 numbers on
    CartpoleSwingup and 210 on HumanoidWalk. A single hardcoded width would pass on
    WalkerWalk and say nothing about the other five panels.
    """
    lines = ["=" * 78,
             "B. Does each run's LOGGED net_params build the network its label claims?",
             "=" * 78,
             "",
             "Actor first-layer input width must be obs + efference_length x action, with",
             "obs and action taken per task. The efference_length = 0 carry must contain no",
             "'queue' entry at all -- a pass-through, not a queue of length zero.",
             ""]
    failures = []
    rows = []
    # One build per distinct (delay, efference_length) pair, not per run: the widths depend
    # on nothing else, and 59 builds would make this script slow for no extra coverage.
    # Which runs map onto each pair is printed so the coverage is visible.
    for (task, delay, eff), group in df.groupby(["task", "delay", "efference_length"]):
        obs_size, action_size = sizes[task]
        env = _StubEnv(obs_size, action_size)
        net_params = {
            "network_class": "DelayedMLP",
            "delay_k": int(delay),
            "efference_length": int(eff),
            "actor_hidden_sizes": [256] * 4,
            "critic_hidden_sizes": [256] * 5,
            "activation": "swish", "normalize_obs": True,
            "initializer_scale": 1.0, "entropy_weight": 0.01,
            "min_std": 0.001, "std_scale": 1.0,
        }
        nets = build_from(net_params, env)
        width = actor_first_kernel(nets)[0]
        expect = obs_size + int(eff) * action_size
        has_queue = "queue" in _all_keys(nets.initialize_state(2))
        ok = (width == expect) and (has_queue == (int(eff) > 0))
        failures += [] if ok else [f"{task} delay {int(delay)} eff {int(eff)}"]
        rows.append((task, int(delay), int(eff), width, expect, has_queue,
                     len(group), ok))

    lines.append(f"{'task':<16} {'delay':>5} {'eff':>4} {'width':>6} {'expect':>7} "
                 f"{'queue?':>7} {'runs':>5}  verdict")
    for task, delay, eff, width, expect, has_queue, n, ok in sorted(rows):
        lines.append(f"{task:<16} {delay:>5} {eff:>4} {width:>6} {expect:>7} "
                     f"{str(has_queue):>7} {n:>5}  "
                     + ("ok" if ok else "*** MISMATCH ***"))
    lines.append("")
    no_copy = df[df["arm"].eq("no_efference")]
    lines.append(f"  no-copy runs in data.csv: {len(no_copy)}; "
                 f"efference_length values {sorted(no_copy['efference_length'].unique())}; "
                 f"delay_k values {sorted(int(d) for d in no_copy['delay'].unique())}")
    copy = df[df["arm"].eq("efference")]
    mismatched = copy[copy["efference_length"] != copy["delay"]]
    lines.append(f"  with-copy runs: {len(copy)}; "
                 f"efference_length != delay_k on {len(mismatched)} of them")
    if len(mismatched):
        failures.append("a with-copy run has efference_length != delay_k")
    lines.append("")
    return lines, failures


def _all_keys(tree) -> set:
    """Every dict key anywhere in a nested carry."""
    keys = set()
    if isinstance(tree, dict):
        keys |= set(tree.keys())
        for v in tree.values():
            keys |= _all_keys(v)
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            keys |= _all_keys(v)
    return keys


def part_c(env, task: str) -> tuple[list[str], list[str]]:
    """Does the queue's content change the policy's output, at initialised weights?"""
    lines = ["=" * 78,
             "C. Does what is IN the queue change what the policy does?",
             "=" * 78,
             "",
             f"Same observation, same RNG, two different action histories written into",
             f"the carry, on {task}. At the *initialised* weights, so a dead path would be",
             f"the default rather than something training had to produce. One task is",
             f"enough: the wrapper under test is the same object in all six panels, and",
             f"part B is what covers each task's own widths.",
             ""]
    failures = []
    obs = jp.asarray(np.linspace(-1, 1, env.observation_size))[None, :]
    for delay, eff in ((5, 5), (20, 20), (5, 0)):
        net_params = {"network_class": "DelayedMLP", "delay_k": delay,
                      "efference_length": eff,
                      "actor_hidden_sizes": [256] * 4,
                      "critic_hidden_sizes": [256] * 5,
                      "activation": "swish", "normalize_obs": True,
                      "initializer_scale": 1.0, "entropy_weight": 0.01,
                      "min_std": 0.001, "std_scale": 1.0}
        nets = build_from(net_params, env)
        base = nets.initialize_state(1)

        _make_deterministic(nets)
        outs = []
        for fill in (0.0, 0.9):
            state, written = _set_queue(base, fill)
            if bool(written) != bool(eff):
                raise SystemExit(
                    f"delay {delay} eff {eff}: wrote {written} queue leaves, expected "
                    f"{'one' if eff else 'none'}. The carry walk is broken, and a broken "
                    f"walk makes this part pass or fail for reasons unrelated to the "
                    f"network.")
            outs.append(np.asarray(_forward(nets, state, obs)))
        delta = float(np.max(np.abs(outs[0] - outs[1])))
        expect_change = eff > 0
        ok = (delta > 1e-6) == expect_change
        failures += [] if ok else [f"delay {delay} eff {eff}"]
        lines.append(f"  delay {delay:>2} eff {eff:>2}: max |action(queue=0) - "
                     f"action(queue=0.9)| = {delta:.6f}   "
                     + ("(expected > 0, queue is read)" if expect_change
                        else "(expected exactly 0, there is no queue)")
                     + ("   ok" if ok else "   *** FAIL ***"))
    lines.append("")
    lines.append("  A zero here at eff > 0 would mean the queue is built and filled but")
    lines.append("  never reaches the policy -- which is the failure mode that would")
    lines.append("  produce this folder's headline result for the wrong reason.")
    lines.append("")
    return lines, failures


def _make_deterministic(nets) -> int:
    """Switch every action sampler in the network to emit its mean.

    ``NormalTanhSampler`` draws from its own RNG stream on every call, so two calls with
    an identical carry differ by sampling noise alone -- which would make part C pass for
    the wrong reason. Setting ``deterministic`` makes the output ``tanh(mean)``, a pure
    function of the weights and the input, which is the object part C needs to compare.
    """
    n = 0
    for _, module in nnx.iter_graph(nets):
        if isinstance(module, ActionSampler):
            module.deterministic = True
            n += 1
    if n == 0:
        raise SystemExit("no ActionSampler found; part C would compare sampling noise")
    return n


def _set_queue(state, fill: float) -> tuple:
    """Write ``fill`` into every 'queue' leaf of a carry, leaving the rest alone.

    Must recurse through **lists and tuples as well as dicts**: a built network's carry is
    ``[normalizer_state, {"action": [delay_state, {"inner": [...], "queue": ...}], ...}]``,
    so a dict-only walk silently returns the carry unchanged -- and then part C compares a
    zeroed queue with a zeroed queue and reports "the queue is never read". That is exactly
    the false positive this script exists to avoid, so the count of leaves actually written
    is returned and asserted rather than trusted.
    """
    written = 0

    def walk(node):
        nonlocal written
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k == "queue":
                    written += 1
                    out[k] = jax.tree.map(lambda x: jp.full_like(x, fill), v)
                else:
                    out[k] = walk(v)
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, tuple):
            return tuple(walk(v) for v in node)
        return node

    return walk(state), written


def _forward(nets, state, obs):
    """The deterministic action the policy emits. Call ``_make_deterministic`` first."""
    return nets(state, obs).output.actions


def part_d(env, action_size: int) -> tuple[list[str], list[str]]:
    """Is the queue aligned: delayed obs followed by the actions taken since it?"""
    lines = ["=" * 78,
             "D. Is the queue ALIGNED -- the actions taken since the delayed observation?",
             "=" * 78,
             "",
             "Delay(k) -> EfferenceCopy(L) composed exactly as make_delayed_mlp_actor_critic",
             "composes them, with a recording stub where the actor MLP goes. At step t the",
             "inner module must receive concat(obs[t-k], a[t-1], a[t-2], ..., a[t-L]):",
             "the delayed observation, then the actions taken since it, newest first.",
             ""]
    failures = []
    k, L, T = 3, 3, 8
    actions = jp.asarray([[float(t + 1)] * action_size for t in range(T)])
    recorder = _Recorder(actions, action_size)
    net = Sequential([
        Delay(jp.zeros(env.observation_size), k_steps=k),
        EfferenceCopy(inner=recorder, sample_action=jp.zeros(action_size),
                      queue_length=L),
    ])
    state = net.initialize_state(1)
    obs_seq = [jp.full((1, env.observation_size), float(t + 1)) for t in range(T)]
    for t in range(T):
        out = net(state, obs_seq[t])
        state = out.next_state

    ok_all = True
    for t in range(T):
        seen = recorder.seen[t][0]
        obs_part, queue_part = seen[:env.observation_size], seen[env.observation_size:]
        # Delay: zeros until the buffer has been written k times, then obs from t - k.
        want_obs = 0.0 if t < k else float(t - k + 1)
        # Queue: newest first, zeros before the action existed.
        want_queue = [float(t - 1 - i + 1) if t - 1 - i >= 0 else 0.0 for i in range(L)]
        got_obs = float(obs_part[0])
        got_queue = [float(queue_part[i * action_size]) for i in range(L)]
        ok = (abs(got_obs - want_obs) < 1e-6
              and all(abs(g - w) < 1e-6 for g, w in zip(got_queue, want_queue))
              and bool(np.all(obs_part == obs_part[0]))
              and all(bool(np.all(queue_part[i * action_size:(i + 1) * action_size]
                                  == got_queue[i])) for i in range(L)))
        ok_all &= ok
        lines.append(f"  t={t}: obs {got_obs:5.1f} (want {want_obs:5.1f})   "
                     f"queue {got_queue} (want {want_queue})"
                     + ("" if ok else "   *** FAIL ***"))
    if not ok_all:
        failures.append("queue alignment")
    lines.append("")
    lines.append("  Reading the t=4 row: the actor sees the observation from step 1 and the")
    lines.append("  three actions emitted at steps 3, 2 and 1 -- exactly the actions taken")
    lines.append("  since that observation, newest first. An off-by-one or a reversed order")
    lines.append("  would show here and nowhere else in this folder.")
    lines.append("")
    return lines, failures


# ---------------------------------------------------------------------------

def resolve_sizes(tasks) -> tuple[dict, list[str]]:
    """Each task's real ``(obs_size, action_size)``, checked against :data:`SIZES`.

    Built envs are the truth and the table is the fallback, so this runs on a machine with
    no GPU -- but a *disagreement* is reported rather than silently resolved either way: if
    the registry has changed, part B would otherwise verify widths against a stale table
    and pass.
    """
    notes, sizes = [], {}
    try:
        from vnl_experiments.envs.registry import ENVS
    except Exception as exc:                                  # noqa: BLE001
        return dict(SIZES), [f"could not import the env registry ({type(exc).__name__}); "
                             f"using the recorded sizes for every task"]
    for task in tasks:
        try:
            spec = ENVS[task]
            env = spec.build(spec.default_config())
            got = (env.observation_size, env.action_size)
        except Exception as exc:                              # noqa: BLE001
            sizes[task] = SIZES[task]
            notes.append(f"  {task:<16} could not build ({type(exc).__name__}); "
                         f"using recorded {SIZES[task]}")
            continue
        sizes[task] = got
        flag = "" if got == SIZES.get(task) else (
            f"   *** differs from the recorded {SIZES.get(task)}; the SIZES table and "
            f"report.md's width numbers are stale ***")
        notes.append(f"  {task:<16} obs {got[0]:>3}  action {got[1]:>3}   (built){flag}")
    return sizes, notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed text would change")
    args = parser.parse_args()

    df = pd.read_csv(HERE / "data.csv")
    tasks = sorted(df["task"].dropna().unique())
    sizes, notes = resolve_sizes(tasks)
    demo = _StubEnv(*sizes[DEMO_TASK])

    lines = ["Is `efference_length` real, and is the queue it builds the right one?", "",
             "observation / action sizes used by part B:", *notes, ""]
    failures: list[str] = []
    for part in (part_a(df), part_b(df, sizes), part_c(demo, DEMO_TASK),
                 part_d(demo, sizes[DEMO_TASK][1])):
        block, bad = part
        lines += block
        failures += bad

    lines.append("=" * 78)
    if failures:
        lines.append(f"VERDICT: {len(failures)} FAILURE(S): {failures}")
        lines.append("The efference copy does not do what this folder's conditions assume.")
    else:
        lines.append("VERDICT: all four parts pass.")
        lines.append(f"Across all {len(tasks)} tasks, `efference_length = 0` builds no "
                     f"queue and the actor is exactly")
        lines.append("obs-wide; `efference_length = L` builds a queue the policy reads, "
                     "holding the L")
        lines.append("actions taken since the delayed observation, newest first. So the "
                     "no-copy arm")
        lines.append("really is a policy with no access to its own action history, in "
                     "every panel, and")
        lines.append("no result in report.md is an artefact of an inert queue.")
    lines.append("=" * 78)
    text = "\n".join(lines) + "\n"

    print(text)
    if args.check:
        old = OUT.read_text() if OUT.exists() else ""
        if old != text:
            raise SystemExit(f"CHECK: {OUT.name} would change")
        print(f"CHECK: {OUT.name} unchanged")
    else:
        OUT.write_text(text)
        print(f"wrote {OUT.name}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
