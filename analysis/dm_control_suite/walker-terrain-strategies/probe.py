"""Do the walker-terrain policies build an implicit forward model? Layer-wise ridge probes.

    OMP_NUM_THREADS=8 ../../../../.venv/bin/python probe.py          # -> probe.csv
    OMP_NUM_THREADS=8 ../../../../.venv/bin/python probe.py --plot   # figures only

The walker counterpart of ``../../rodent/implicit-forward-model/``. For every run
(``common.load_runs``, 62) and every representation on the actor pathway, a ridge
decoder is fit to the **current** normalised observation (the normaliser output ``0``
at step t) and to its **delta** from the delayed one (obs_t - obs_{t-5}), and scored on
held-out episodes.

Representations (all at step t):

* ``input`` -- exactly what the actor (DelayedMLP) or predictor (FlatForwardModel) sees:
  the delayed normalised obs plus the efference queue rebuilt from recorded actions.
  The **linear-input baseline**: a layer that beats it has computed something.
* ``input_rf`` -- ridge on 2048 fixed random swish features of the standardised input:
  a rough ceiling on what a nonlinear readout of the input affords.
* ``act0..act3`` (actor / FM decoder hidden layers), and for FM ``pred0..pred3`` and
  ``pred_out`` (the explicit prediction), each **trained** (recorded) and **init**
  (the same layer recomputed from the run's rebuilt initial weights on the same input).

Targets and groups: proprio dims 0-23 (``all``), ``orient`` 0-14, ``vel`` 15-23, and for
sensor runs ``sensor`` 24-27 (block distance / top of the next two blocks). R^2 is per
dim on the concatenated held-out predictions, then averaged uniformly within a group;
dims with (near-)zero variance in the scored subset are skipped (the sensor tops on the
flat course).

CV: 3 outer folds, fold j = episode index j of every course set (so 3 episodes of 960
steps held out). Ridge alpha is chosen per (run, representation, target) by inner
leave-one-episode-index-out CV on the two training folds. Features are standardised
with whole-run statistics (unsupervised; no target information) so that the Gram
matrices of the three episode groups can be summed. The first 40 steps of each episode
are dropped (stand-up transient, zero-filled delay and efference buffers).

FM runs additionally get ``weights == "explicit"`` rows: the prediction ``pred_out``
scored directly against the true current obs (no regression), overall, per course and
split by ``|blk0.dist| < 0.5 m`` on the two block courses (``near_block`` /
``far_block``); ``block_sensor`` is recorded for blind runs too.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "probe.csv"
SKIP = 40
N_RF = 2048
LAMBDAS = np.logspace(-6, 1, 15)  # alpha = lambda * n_train (features standardised)
MIN_VAR = 1e-6
GROUPS = {"all": list(range(24)), "orient": list(range(15)), "vel": list(range(15, 24)),
          "sensor": list(range(24, 28))}
NEAR = 0.5
LAYERS = ("pred0", "pred1", "pred2", "pred3", "pred_out", "act0", "act1", "act2", "act3")


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------

def run_data(r) -> dict:
    """Stacked [rows, ...] arrays over the 3 course sets x 3 episodes, t >= SKIP."""
    arrays, cfg = C.load_weights(r.wandb_id)
    eff = int(cfg["net_params"]["efference_length"])
    fm = bool(r.forward_model)
    init_stacks = C.mlp_stacks(C.init_arrays(r.wandb_id, int(r.seed)))
    reps: dict[str, list] = {}
    cur, dly, course, ep, blk = [], [], [], [], []
    for cs in C.COURSE_SETS:
        rec = C.load_rec(r.wandb_id, cs)
        names = C.layer_names(rec)
        x = C.actor_input(rec, eff)
        obs = rec.acts["0"]
        d = C.delayed(obs, C.DELAY)
        T, N, _ = obs.shape
        sl = slice(SKIP, T)
        flat = lambda a: a[sl].transpose(1, 0, 2).reshape(N * (T - SKIP), -1)  # ep-major
        layers = [k for k in names if k in LAYERS]
        tr = {k: rec.acts[names[k]] for k in layers}
        ini = C.actor_forward(init_stacks, x, fm)
        reps.setdefault("input|input", []).append(flat(x))
        for k in layers:
            reps.setdefault(f"{k}|trained", []).append(flat(tr[k]))
            reps.setdefault(f"{k}|init", []).append(flat(ini[k]))
        cur.append(flat(obs))
        dly.append(flat(d))
        course += [cs] * (N * (T - SKIP))
        ep.append(np.repeat(np.arange(N), T - SKIP))
        blk.append(flat(rec.data["block_sensor"])[:, 0])
    out = {k: np.concatenate(v).astype(np.float64) for k, v in reps.items()}
    return dict(reps=out, cur=np.concatenate(cur).astype(np.float64),
                dly=np.concatenate(dly).astype(np.float64), course=np.array(course),
                ep=np.concatenate(ep), blk0=np.concatenate(blk))


def random_features(x: np.ndarray, n: int = N_RF, seed: int = 0) -> np.ndarray:
    z = (x - x.mean(0)) / (x.std(0) + 1e-6)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.5 / np.sqrt(z.shape[1]), (z.shape[1], n))
    b = rng.normal(0, 1.0, n)
    return C.swish(z @ W + b)


# --------------------------------------------------------------------------------------
# ridge with summed per-group sufficient statistics
# --------------------------------------------------------------------------------------

def _r2_dims(y, p):  # inner-CV alpha selection
    sst = ((y - y.mean(0)) ** 2).sum(0)
    sse = ((y - p) ** 2).sum(0)
    ok = sst / len(y) > MIN_VAR
    r2 = np.full(y.shape[1], np.nan)
    r2[ok] = 1 - sse[ok] / sst[ok]
    return r2


def cv_ridge(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, blocks: dict[str, list]):
    """Held-out predictions for Y; alpha chosen per target block by inner CV.

    ``blocks`` maps a target name to its columns in Y (alpha is selected on the mean R^2
    over those columns).
    """
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    g_ids = np.unique(groups)
    stats = {}
    for g in g_ids:
        m = groups == g
        Xg, Yg = X[m], Y[m]
        stats[g] = (m.sum(), Xg.sum(0), Yg.sum(0), Xg.T @ Xg, Xg.T @ Yg)

    def fit(train):
        n = sum(stats[g][0] for g in train)
        sx = sum(stats[g][1] for g in train)
        sy = sum(stats[g][2] for g in train)
        G = sum(stats[g][3] for g in train)
        XY = sum(stats[g][4] for g in train)
        mx, my = sx / n, sy / n
        G = G - n * np.outer(mx, mx)
        XY = XY - n * np.outer(mx, my)
        s, V = np.linalg.eigh(G)
        s = np.clip(s, 0, None)
        return n, mx, my, s, V, V.T @ XY

    def predict(model, Xte, lam):
        n, mx, my, s, V, W = model
        Z = (Xte - mx) @ V
        return (Z / (s + lam * n)) @ W + my

    pred = np.zeros_like(Y)
    for j in g_ids:
        train = [g for g in g_ids if g != j]
        # inner: leave one of the training groups out
        scores = {b: np.zeros(len(LAMBDAS)) for b in blocks}
        for v in train:
            inner = [g for g in train if g != v]
            model = fit(inner)
            mv = groups == v
            for li, lam in enumerate(LAMBDAS):
                p = predict(model, X[mv], lam)
                r2 = _r2_dims(Y[mv], p)
                for b, cols in blocks.items():
                    scores[b][li] += np.nanmean(r2[cols])
        model = fit(train)
        mt = groups == j
        for b, cols in blocks.items():
            lam = LAMBDAS[int(np.argmax(scores[b]))]
            pred[np.ix_(mt, cols)] = predict(model, X[mt], lam)[:, cols]
    return pred


# --------------------------------------------------------------------------------------
# per run
# --------------------------------------------------------------------------------------

def group_scores(cur, dly, pcur, sensor: bool) -> dict[tuple[str, str], float]:
    """``{(target, group): R^2}`` of a current-obs prediction ``pcur``.

    ``current``: the usual R^2 against the current obs. ``delta``: the *same* prediction
    error, normalised by the variance of obs_t - obs_{t-5} instead -- i.e. the fraction of
    the last 125 ms of change the representation recovers. For the input (which contains
    the delayed obs linearly) this equals a ridge regression of the delta; for a deep
    layer it does not require the layer to also carry the delayed obs; and it is the
    scale on which the explicit prediction is scored directly.
    """
    d = cur - dly
    out = {}
    for tgt, ref in (("current", cur), ("delta", d)):
        sst = ((ref - ref.mean(0)) ** 2).sum(0)
        sse = ((cur - pcur) ** 2).sum(0)
        ok = sst / len(ref) > MIN_VAR
        r2 = np.where(ok, 1 - sse / np.where(ok, sst, 1), np.nan)
        for g, cols in GROUPS.items():
            if g == "sensor" and not sensor:
                continue
            v = r2[cols]
            out[(tgt, g)] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    return out


def probe_run(r) -> list[dict]:
    D = run_data(r)
    sensor = bool(r.sensor)
    nd = 28 if sensor else 24
    cur, dly = D["cur"][:, :nd], D["dly"][:, :nd]
    blocks = {"current": list(range(nd))}
    reps = dict(D["reps"])
    reps["input|input_rf"] = random_features(reps["input|input"])
    base = dict(wandb_id=r.wandb_id, condition=r.condition, arch=r.arch,
                **{f: bool(r[f]) for f in C.FACTORS}, seed=int(r.seed),
                reward_480M=float(r.reward_480M))
    rows = []
    preds = {}

    def emit(layer, weights, course, m, P):
        for (tgt, g), v in group_scores(cur[m], dly[m], P[m], sensor).items():
            rows.append(dict(base, layer=layer, weights=weights, target=tgt, group=g,
                             course=course, r2=v))

    allm = np.ones(len(cur), bool)
    for key, X in reps.items():
        layer, weights = key.split("|")
        preds[key] = cv_ridge(X, cur, D["ep"], blocks)
        emit(layer, weights, "all", allm, preds[key])
    act_keys = [k for k in reps if k.startswith("act") and k.endswith("|trained")]
    delta_all = {k: group_scores(cur, dly, preds[k], sensor)[("delta", "all")]
                 for k in act_keys}
    best = max(act_keys, key=delta_all.get)
    keys = ["input|input", "input|input_rf", best]
    if r.forward_model:
        keys.append("pred_out|trained")
        preds["pred_out_direct|explicit"] = reps["pred_out|trained"][:, :nd]
        keys.append("pred_out_direct|explicit")
        emit("pred_out_direct", "explicit", "all", allm, preds["pred_out_direct|explicit"])
    block_course = np.isin(D["course"], ["h0.3_irr1", "stairs_h0.3"])
    near = np.abs(D["blk0"]) < NEAR
    subsets = {**{cs: D["course"] == cs for cs in C.COURSE_SETS},
               "near_block": block_course & near, "far_block": block_course & ~near}
    for key in keys:
        layer, weights = key.split("|")
        if key == best:
            layer = "best_act:" + layer
        for name, m in subsets.items():
            emit(layer, weights, name, m, preds[key])
    return rows


def main_extract(only=None):
    runs = C.load_runs()
    if only:
        runs = runs[runs.wandb_id.isin(only)]
    rows = []
    t0 = time.time()
    for i, r in runs.iterrows():
        rows += probe_run(r)
        print(f"[{i + 1}/{len(runs)}] {r.wandb_id} {r.condition} {time.time() - t0:.0f}s",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT if not only else HERE / "probe_test.csv", index=False,
              float_format="%.5f")
    return df


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------

FAMILIES = [("MLP, no efference", False, False), ("MLP + efference", False, True),
            ("FM, no efference", True, False), ("FM + efference", True, True)]


def _layer_order(fm: bool):
    return (["pred0", "pred1", "pred2", "pred3", "pred_out"] if fm else []) + \
        ["act0", "act1", "act2", "act3"]


def plot(df: pd.DataFrame, target: str = "delta"):
    import matplotlib.pyplot as plt

    from vnl_experiments.wandb_utils import style
    style.apply_style()
    fig_dir = HERE / "figures"
    a = df[(df.target == target) & (df.group == "all") & (df.course == "all")]

    # (1) layer profiles per condition, faceted by family
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), sharey=True)
    for ax, (title, fm, eff) in zip(axes, FAMILIES):
        sub = a[(a.forward_model == fm) & (a.efference == eff)]
        order = _layer_order(fm)
        xs = np.arange(len(order) + 2)
        labels = ["input", "input_rf"] + order
        conds = sorted(sub.condition.unique())
        cmap = plt.get_cmap("tab10")
        for ci, cond in enumerate(conds):
            c = sub[sub.condition == cond]
            col = cmap(ci % 10)
            for wts, ls in (("trained", "-"), ("init", "--")):
                m = c[c.weights == wts].groupby("layer").r2.mean().reindex(order)
                ax.plot(xs[2:], m.values, ls, color=col, marker="o" if wts == "trained"
                        else None, ms=4, lw=1.6 if wts == "trained" else 1.0,
                        label=f"{cond} (n={c.wandb_id.nunique()})" if wts == "trained"
                        else None)
            inp = c[c.weights == "input"].r2.mean()
            rf = c[c.weights == "input_rf"].r2.mean()
            ax.plot([0, 1], [inp, rf], "s", color=col, ms=6)
            ax.axhline(inp, color=col, lw=0.6, alpha=0.5)
            # per-run dots for trained layers
            for wid, cr in c[c.weights == "trained"].groupby("wandb_id"):
                ax.plot(xs[2:], cr.set_index("layer").r2.reindex(order).values, ".",
                        color=col, alpha=0.35, ms=3)
        ax.set_xticks(xs, labels, rotation=45, ha="right")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel(f"held-out R² ({target}: obs$_t$ − obs$_{{t-5}}$)" if target == "delta"
                       else "held-out R² (current obs$_t$)")
    fig.suptitle(f"Decoding the {'delta' if target == 'delta' else 'current'} proprioception"
                 f" along the actor pathway (solid trained, dashed init; squares = "
                 f"input linear / random-features ceiling); delay 5 = 125 ms", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / f"probe_layers_{target}.png", dpi=150)
    plt.close(fig)

    # (2) summary: best actor layer minus input baseline, per run
    rows = []
    for wid, g in a.groupby("wandb_id"):
        r0 = g.iloc[0]
        inp = g[g.weights == "input"].r2.iloc[0]
        rf = g[g.weights == "input_rf"].r2.iloc[0]
        tr = g[(g.weights == "trained") & g.layer.str.startswith("act")]
        it = g[(g.weights == "init") & g.layer.str.startswith("act")]
        po = g[(g.weights == "trained") & (g.layer == "pred_out")].r2
        rows.append(dict(wandb_id=wid, condition=r0.condition, seed=r0.seed,
                         fm=r0.forward_model, eff=r0.efference,
                         best_trained=tr.r2.max() - inp, best_init=it.r2.max() - inp,
                         rf=rf - inp, pred_out=(po.iloc[0] - inp) if len(po) else np.nan))
    s = pd.DataFrame(rows)
    s["fam"] = s.fm.map({False: "MLP", True: "FM"}) + s.eff.map({False: "", True: "+eff"})
    s = s.sort_values(["fm", "eff", "condition"])
    conds = list(dict.fromkeys(s.condition))
    fig, ax = plt.subplots(figsize=(13, 4.8))
    seeds = sorted(s.seed.unique())
    mk = dict(zip(seeds, "osD^vP<>"))
    for i, cond in enumerate(conds):
        c = s[s.condition == cond]
        jit = np.linspace(-0.15, 0.15, len(c)) if len(c) > 1 else [0]
        for (_, rr), dx in zip(c.iterrows(), jit):
            ax.plot(i + dx, rr.best_trained, mk[rr.seed], color="C0", ms=6)
            ax.plot(i + dx, rr.best_init, mk[rr.seed], color="0.6", ms=5, mfc="none")
            ax.plot(i + dx, rr.rf, mk[rr.seed], color="C3", ms=5, mfc="none")
            if np.isfinite(rr.pred_out):
                ax.plot(i + dx, rr.pred_out, mk[rr.seed], color="C2", ms=6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(conds)), conds, rotation=40, ha="right")
    ax.set_ylabel(f"R² − linear-input R² ({target}, all proprio)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="C0", marker="o", ls="", label="best actor/decoder layer, trained"),
         Line2D([], [], color="0.6", marker="o", mfc="none", ls="", label="best actor layer, init"),
         Line2D([], [], color="C3", marker="o", mfc="none", ls="", label="random-features ceiling"),
         Line2D([], [], color="C2", marker="o", ls="", label="FM predictor output (pred_out)")]
    h += [Line2D([], [], color="0.3", marker=mk[sd], ls="", label=f"seed i{sd}") for sd in seeds]
    ax.legend(handles=h, fontsize=7, ncol=2, loc="best")
    ax.set_title(f"Gain over the linear-input baseline, per run ({target}); 0 = no better than "
                 f"a linear readout of [delayed obs + efference]", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / f"probe_summary_{target}.png", dpi=150)
    plt.close(fig)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    df = pd.read_csv(OUT) if args.plot else main_extract(args.only)
    if not args.only:
        for t in ("delta", "current"):
            plot(df, t)
