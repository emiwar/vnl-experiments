"""Shared loaders for walker-terrain-strategies: runs, recordings, weights, forward pass.

The cohort is ``../walker-terrain-factors/data.csv`` (selection, factors, comparability
and readouts all live there), restricted to the 62 ``complete`` runs, every one of which
has a ``weights`` and three ``activations_dmc`` artifacts at the 480 215 040 checkpoint.

Everything here is numpy. The actor of both architectures is feed-forward from its
input, so given the recorded network input (the normaliser output ``0``, delayed by
``delay_k`` steps, plus the efference queue rebuilt from the recorded actions) the
actor can be re-run offline: :func:`actor_forward` does that and
``check_forward.py`` asserts it reproduces the recorded activations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FACTORS_DIR = HERE.parent / "walker-terrain-factors"
REPO = HERE.parents[2]
ART = REPO / "artifacts"
STEP = 480215040
WEIGHTS_SPEC = "w-s0480215040-70e4fc12"
COURSE_SETS = ("flat", "h0.3_irr1", "stairs_h0.3")
DELAY = 5
FACTORS = ["efference", "forward_model", "sensor", "privileged_critic", "servo"]

# Observation layout (WalkerTerrain._get_obs): 7 bodies x (xx, xz) of xmat, torso zz,
# 9 qvel, then [dist0, top0, dist1, top1] of the block sensor when observed.
BODIES = ["torso", "r_thigh", "r_leg", "r_foot", "l_thigh", "l_leg", "l_foot"]
QVEL = ["vz", "vx", "wy", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle"]
OBS_NAMES = ([f"{b}.{c}" for b in BODIES for c in ("xx", "xz")] + ["torso.zz"]
             + [f"qvel.{q}" for q in QVEL])
SENSOR_NAMES = ["blk0.dist", "blk0.top", "blk1.dist", "blk1.top"]
OBS_GROUPS = {"orient": slice(0, 15), "vel": slice(15, 24), "sensor": slice(24, 28)}
JOINTS = ["r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle"]
ACTION_SIZE = 6


def load_runs() -> pd.DataFrame:
    d = pd.read_csv(FACTORS_DIR / "data.csv")
    d = d[d.complete].copy()
    d["sensor"] = ~d["blind"]
    d["seed"] = d["seed_pair"].str.extract(r"i(\d+)/").astype(int)[0]
    d["arch"] = d.apply(arch_key, axis=1)
    return d.reset_index(drop=True)


def arch_key(r) -> str:
    """Runs with the same key have identically shaped weights (names aside)."""
    net = "FM" if r["forward_model"] else "MLP"
    eff = "eff" if r["efference"] else "noeff"
    sen = "sensor" if r["sensor"] else "blind"
    crit = ""
    if not r["forward_model"] and r["efference"]:
        crit = "+privC" if r["privileged_critic"] else "+delayedC"
    return f"{net}/{eff}/{sen}{crit}"


# --------------------------------------------------------------------------------------
# recordings
# --------------------------------------------------------------------------------------

@dataclass
class Rec:
    wandb_id: str
    course: str
    attrs: dict
    data: dict = field(default_factory=dict)
    acts: dict = field(default_factory=dict)

    @property
    def T(self):
        return self.data["reward"].shape[0]


def rec_path(wandb_id: str, course: str) -> Path:
    hits = list((ART / "activations_dmc" / wandb_id).glob(f"dact-{course}-s{STEP:010d}-*.h5"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{wandb_id} {course}: {hits}")
    return hits[0]


def load_rec(wandb_id: str, course: str, acts: bool = True) -> Rec:
    with h5py.File(rec_path(wandb_id, course)) as f:
        attrs = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in f.attrs.items()}
        data = {k: f[k][()] for k in f if k != "activations"}
        a = {}
        if acts:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    a[name] = obj[()].astype(np.float32)
            f["activations"].visititems(visit)
    return Rec(wandb_id, course, attrs, data, a)


def layer_names(rec: Rec) -> dict[str, str]:
    """Canonical role -> recorded activation name, for both architectures and layouts."""
    names = list(rec.acts)
    fm = rec.attrs["network_class"] == "FlatForwardModel"
    out = {"norm_obs": "0"}
    if fm:
        p = "1/action/1/"
        out.update({"delayed": p + "delay"})
        out.update({f"pred{i}": p + f"predictor/{i}" for i in range(4)})
        out["pred_out"] = p + "predictor/4"
        out.update({f"act{i}": p + f"decoder/{i}" for i in range(4)})
        out["head"] = p + "decoder/4"
        out["action"] = p + "decoder/5/action"
        vpre = "1/value/"
    elif "1/action/0" in names:  # privileged critic (Parallel actor/critic)
        p = "1/action/1/"
        out["delayed"] = "1/action/0"
        out.update({f"act{i}": p + f"{i}" for i in range(4)})
        out["head"] = p + "4"
        out["action"] = p + "5/action"
        vpre = "1/value/"
    else:  # delayed critic: Sequential(norm, delay, efference(actor+critic))
        out["delayed"] = "1"
        out.update({f"act{i}": f"2/action/{i}" for i in range(4)})
        out["head"] = "2/action/4"
        out["action"] = "2/action/5/action"
        vpre = "2/value/"
    out.update({f"val{i}": vpre + f"{i}" for i in range(5)})
    out["value"] = vpre + "5"
    missing = [v for v in out.values() if v not in rec.acts]
    if missing:
        raise KeyError(f"{rec.wandb_id}: missing {missing} in {names}")
    return out


def efference_queue(actions: np.ndarray, length: int) -> np.ndarray:
    """``[T, N, L*A]`` newest-first queue of the actions emitted *before* step t."""
    T, N, A = actions.shape
    out = np.zeros((T, N, length * A), np.float32)
    for j in range(length):
        lag = j + 1
        out[lag:, :, j * A:(j + 1) * A] = actions[:-lag]
    return out


def delayed(x: np.ndarray, k: int) -> np.ndarray:
    """``x`` delayed by ``k`` steps along axis 0, zero-filled (the Delay layer's init)."""
    out = np.zeros_like(x)
    out[k:] = x[:-k] if k else x
    return out


# --------------------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------------------

def load_weights(wandb_id: str) -> tuple[dict[str, np.ndarray], dict]:
    z = np.load(ART / "weights" / wandb_id / f"{WEIGHTS_SPEC}.npz", allow_pickle=False)
    arrays = {k: z[k] for k in z.files if k != "__config__"}
    config = json.loads(str(z["__config__"]))
    return arrays, config


def mlp_stacks(arrays: dict[str, np.ndarray]) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """``{"actor"|"predictor"|"critic": [(W, b), ...]}`` in layer order."""
    stacks: dict[str, dict[int, dict]] = {}
    for k, v in arrays.items():
        if not k.endswith(("linear/kernel", "linear/bias")):
            continue
        parts = k.split("/")
        i = int(parts[parts.index("linear") - 1])
        if "predictor" in parts:
            role = "predictor"
        elif "value" in parts:
            role = "critic"
        else:
            role = "actor"
        stacks.setdefault(role, {}).setdefault(i, {})[parts[-1]] = v
    return {r: [(d[i]["kernel"], d[i]["bias"]) for i in sorted(d)] for r, d in stacks.items()}


def swish(x):
    return x / (1.0 + np.exp(-x))


def mlp_forward(stack, x, return_all=False):
    hs = []
    for i, (W, b) in enumerate(stack):
        x = x @ W + b
        if i < len(stack) - 1:
            x = swish(x)
        hs.append(x)
    return hs if return_all else x


def actor_input(rec: Rec, eff_len: int, k: int = DELAY) -> np.ndarray:
    """The actor's (or predictor's) input at each step: delayed norm obs + efference."""
    d = delayed(rec.acts["0"], k)
    if eff_len:
        d = np.concatenate([d, efference_queue(rec.data["actions"], eff_len)], axis=-1)
    return d


def actor_forward(stacks, x, fm: bool):
    """All actor-path activations from the actor input ``x``; mirrors :func:`layer_names`."""
    out = {}
    if fm:
        ph = mlp_forward(stacks["predictor"], x, return_all=True)
        out.update({f"pred{i}": h for i, h in enumerate(ph[:-1])})
        out["pred_out"] = ph[-1]
        x = ph[-1]
    ah = mlp_forward(stacks["actor"], x, return_all=True)
    out.update({f"act{i}": h for i, h in enumerate(ah[:-1])})
    out["head"] = ah[-1]
    out["action"] = np.tanh(ah[-1][..., :ACTION_SIZE])
    return out


@lru_cache(maxsize=None)
def init_arrays(wandb_id: str, seed: int) -> dict[str, np.ndarray]:
    """The run's weights at initialisation, rebuilt as train.py builds them.

    ``seed`` is the run's top-level (init) seed -- the ``i`` of ``seed_pair`` -- which
    the weights file's config does not carry. ``train.py`` calls
    ``build_network(net_params, env, nnx.Rngs(seed))`` with a fresh ``Rngs``.
    """
    from flax import nnx

    from vnl_experiments.delays.network_builders import build_network

    _, config = load_weights(wandb_id)
    net_params = config["net_params"]
    env = _ShapeEnv(observation_size=24 + 4 * bool(config["env_params"]["terrain"]["observed"]),
                    action_size=ACTION_SIZE)
    nets = build_network(dict(net_params), env, nnx.Rngs(seed))
    return _state_arrays(nets)


class _ShapeEnv:
    def __init__(self, observation_size, action_size):
        self.observation_size = observation_size
        self.action_size = action_size


def _state_arrays(nets) -> dict[str, np.ndarray]:
    from flax import nnx

    from vnl_experiments.artifacts.weights import flat_key

    _, _, state = nnx.split(nets, nnx.RngKey, ...)
    return {flat_key(path): np.asarray(var.get_value())
            for path, var in nnx.to_flat_state(state)}
