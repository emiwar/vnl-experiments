"""What each policy's action is sensitive to, on the states it actually visits.

    OMP_NUM_THREADS=8 ../../../../.venv/bin/python attribution.py

For every run, the Jacobian of the action mean (pre-tanh) with respect to the actor
pathway's *input* -- delayed normalised obs [+ block sensor] [+ efference queue] -- is
taken at every ``STRIDE``-th step of all nine recorded episodes (after ``gait.SKIP``).
For FlatForwardModel runs the Jacobian runs through predictor and decoder, so it is the
same quantity: sensitivity of the action to what the network is given.

Attribution of input i = mean over states and over the 6 actions of |dA_j/dx_i| *
std(x_i), the change in action from a typical variation of that input. It is reported
as each input group's share of the run's total: ``orient`` (15 body orientations +
torso zz), ``vel`` (9 qvel), ``sensor`` (4 block-sensor dims), ``eff_lag1..5`` (the six
actions emitted 1..5 steps ago). Absolute totals are kept too (``total``).

For FM runs the decoder alone is also attributed with respect to the *predicted* current
obs (``fm_decoder`` rows), which says what the decoder reads from the forward model.

Internal efference loop. With an efference copy the policy is a recurrent system on
its own actions, ``a_t = tanh(mu(o_{t-5}, a_{t-1}, ..., a_{t-5}))``. Holding the sensory
input fixed, its linearisation at each visited state is the 30x30 companion matrix
whose top block is ``diag(1 - a_t^2) @ dA/d(efference)`` and whose rest shifts the
queue. Its spectral radius ``rho`` says whether the action->efference->action loop on its
own decays (< 1), or would sustain/grow an oscillation (>= 1, a CPG-like internal
rhythm), and the dominant eigenvalue's angle its frequency. Per run, ``loop.csv`` keeps
the median rho, the fraction of states with rho >= 1, and the median frequency (Hz) of
the dominant eigenvalue.

Writes ``attribution.csv`` (long: run x group x scope), ``attribution_inputs.csv``
(per individual input, for the figures) and ``loop.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
from gait import SKIP  # noqa: E402

STRIDE = 4


def _jax_mlp(stack):
    import jax.numpy as jnp
    params = [(jnp.asarray(W), jnp.asarray(b)) for W, b in stack]

    def f(x):
        for i, (W, b) in enumerate(params):
            x = x @ W + b
            if i < len(params) - 1:
                x = x * (1.0 / (1.0 + jnp.exp(-x)))
        return x
    return f


def input_names(n_obs: int, eff_len: int) -> list[str]:
    names = list(C.OBS_NAMES) + (C.SENSOR_NAMES if n_obs == 28 else [])
    names += [f"eff{lag}.{j}" for lag in range(1, eff_len + 1) for j in C.JOINTS]
    return names


def group_of(name: str) -> str:
    if name.startswith("eff"):
        return f"eff_lag{name[3]}"
    if name.startswith("blk"):
        return "sensor"
    if name.startswith("qvel"):
        return "vel"
    return "orient"


def attribute(jac: np.ndarray, x: np.ndarray, names: list[str]) -> pd.Series:
    """Per-input |J| * std(x), averaged over states and actions."""
    s = np.abs(jac).mean(axis=(0, 1)) * x.std(axis=0)
    return pd.Series(s, index=names)


def loop_stats(jac: np.ndarray, x: np.ndarray, a: np.ndarray, n_obs: int,
               eff: int) -> dict:
    """Spectral radius and frequency of the linearised efference loop, per state."""
    A = C.ACTION_SIZE
    je = jac[:, :, n_obs:] * (1.0 - a[:, :, None] ** 2)            # [N, 6, 6*eff]
    n = je.shape[0]
    M = np.zeros((n, A * eff, A * eff), np.float64)
    M[:, :A, :] = je
    for j in range(1, eff):
        M[:, j * A:(j + 1) * A, (j - 1) * A:j * A] = np.eye(A)
    ev = np.linalg.eigvals(M)
    k = np.abs(ev).argmax(axis=1)
    dom = ev[np.arange(n), k]
    rho = np.abs(dom)
    freq = np.abs(np.angle(dom)) / (2 * np.pi * 0.025)
    return dict(rho_median=float(np.median(rho)), rho_p90=float(np.percentile(rho, 90)),
                frac_rho_ge1=float((rho >= 1).mean()),
                freq_median=float(np.median(freq)),
                frac_oscillatory=float((np.abs(np.angle(dom)) > 1e-3).mean()))


def main():
    import jax

    runs = C.load_runs()
    rows, per_input, loops = [], [], []
    for _, r in runs.iterrows():
        arrays, cfg = C.load_weights(r.wandb_id)
        st = C.mlp_stacks(arrays)
        eff = int(cfg["net_params"]["efference_length"])
        xs, preds, acts = [], [], []
        for course in C.COURSE_SETS:
            rec = C.load_rec(r.wandb_id, course)
            x = C.actor_input(rec, eff)[SKIP::STRIDE]
            xs.append(x.reshape(-1, x.shape[-1]))
            acts.append(rec.data["actions"][SKIP::STRIDE].reshape(-1, C.ACTION_SIZE))
            if r.forward_model:
                p = rec.acts[C.layer_names(rec)["pred_out"]][SKIP::STRIDE]
                preds.append(p.reshape(-1, p.shape[-1]))
        x = np.concatenate(xs).astype(np.float32)
        n_obs = x.shape[1] - eff * C.ACTION_SIZE
        names = input_names(n_obs, eff)

        actor = _jax_mlp(st["actor"])
        if r.forward_model:
            pred = _jax_mlp(st["predictor"])
            f = lambda v: actor(pred(v))[: C.ACTION_SIZE]  # noqa: E731
        else:
            f = lambda v: actor(v)[: C.ACTION_SIZE]  # noqa: E731
        jac = np.asarray(jax.vmap(jax.jacrev(f))(x))                 # [N, 6, D]
        scopes = [("policy", attribute(jac, x, names))]
        if eff:
            loops.append(dict(wandb_id=r.wandb_id,
                              **loop_stats(jac, x, np.concatenate(acts), n_obs, eff)))
        if r.forward_model:
            p = np.concatenate(preds).astype(np.float32)
            dj = np.asarray(jax.vmap(jax.jacrev(lambda v: actor(v)[: C.ACTION_SIZE]))(p))
            scopes.append(("fm_decoder", attribute(dj, p, names[:n_obs])))
        for scope, s in scopes:
            total = float(s.sum())
            for grp, v in s.groupby(s.index.map(group_of)).sum().items():
                rows.append(dict(wandb_id=r.wandb_id, scope=scope, group=grp,
                                 share=float(v / total), value=float(v), total=total))
            per_input += [dict(wandb_id=r.wandb_id, scope=scope, input=k, value=float(v),
                               share=float(v / total)) for k, v in s.items()]
        print(r.wandb_id, r.condition, flush=True)
    meta = runs[["wandb_id", "condition", "arch", *C.FACTORS, "seed", "reward_480M"]]
    here = Path(__file__).parent
    meta.merge(pd.DataFrame(rows), on="wandb_id").to_csv(
        here / "attribution.csv", index=False, float_format="%.5g")
    meta.merge(pd.DataFrame(per_input), on="wandb_id").to_csv(
        here / "attribution_inputs.csv", index=False, float_format="%.5g")
    lp = meta.merge(pd.DataFrame(loops), on="wandb_id")
    lp.to_csv(here / "loop.csv", index=False, float_format="%.5g")
    print(lp.groupby("condition")[["rho_median", "rho_p90", "frac_rho_ge1", "freq_median",
                                   "frac_oscillatory"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
