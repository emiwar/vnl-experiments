"""What did training change in the weights, and can the factors be read back from them?

    OMP_NUM_THREADS=8 ../../../../.venv/bin/python weights.py

Cohort: the 62 complete runs of ``../walker-terrain-factors/data.csv`` (see ``common.py``),
weights at the 480 215 040 checkpoint. Each run's initial weights are rebuilt exactly as
``train.py`` builds them (``C.init_arrays``), so everything here is about
``dW = W - W0``: ~80 % of a trained hidden matrix is still its initialisation
(``checks.txt``), and raw weights mostly measure the seed.

Layers are named by role and index: ``actor{i}`` (the DelayedMLP actor, or the
FlatForwardModel *decoder*), ``predictor{i}`` (FM only), ``critic{i}``. ``actor1..4`` and
``critic1..5`` have the same shape in all 62 runs ("shape-shared"); ``actor0`` /
``predictor0`` / ``critic0`` have an architecture-dependent fan-in.

Writes
------
* ``weights_features.csv`` -- one row per run: per-layer change/spectrum stats, head
  biases and column norms, first-layer input-row ratios.
* ``weights_similarity.csv`` -- every same-seed pair of runs: cosine of their dW on the
  shape-shared actor and critic layers, whether their init is bit-identical there.
* ``weights_classify.csv`` -- factor-inference results (leave-one-seed-out, perm. null),
  single-feature accuracies, and the matched-reward servo pair tests.
* ``figures/weights_*.png`` and ``weights_notes.md``.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

SCRATCH = Path("/tmp/claude-1000/-home-emil-Development-vnl/"
               "e50acd31-75e5-4056-bab5-7ae5465eeb12/scratchpad/init")
SHARED = [f"actor{i}" for i in range(1, 5)] + [f"critic{i}" for i in range(1, 6)]
N_PERM = 300
RNG = np.random.default_rng(0)
FACTORS = C.FACTORS
MIN_STD = 1e-3

#: identical config + identical seed pair: differ by GPU nondeterminism only.
REPLICATES = [("j70fn2g7", "zz0reqyq"), ("lptmmkvi", "o2zf70hu"), ("5tiufunj", "u10wy89i"),
              ("c5arx0qj", "suzrcsxy"), ("f3eani0z", "h1800i1b"), ("2mxf1piw", "kfld9i6s"),
              ("40w5wtaw", "u8z1oecr"), ("77va16yb", "sog4k5xi")]
#: matched-reward torque/servo cells with the same architecture.
SERVO_PAIRS = [("eff+fm+priv", "eff+fm+servo+priv"),
               ("eff+fm+sensor+priv", "eff+fm+servo+sensor+priv")]


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------

def layers_of(arrays: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    st = C.mlp_stacks(arrays)
    return {f"{role}{i}": wb for role, lst in st.items() for i, wb in enumerate(lst)}


def load_init(wid: str, seed: int) -> dict:
    p = SCRATCH / f"{wid}.npz"
    if p.exists():
        z = np.load(p)
        return {k.replace("|", "/"): z[k] for k in z.files}
    return C.init_arrays(wid, seed)


def load_all(runs):
    W, W0 = {}, {}
    for _, r in runs.iterrows():
        tr, _ = C.load_weights(r.wandb_id)
        W[r.wandb_id] = layers_of(tr)
        W0[r.wandb_id] = layers_of(load_init(r.wandb_id, int(r.seed)))
        assert set(W[r.wandb_id]) == set(W0[r.wandb_id]), r.wandb_id
    return W, W0


def input_names(r) -> list[str]:
    names = list(C.OBS_NAMES) + (list(C.SENSOR_NAMES) if r.sensor else [])
    if r.efference:
        names += [f"eff.lag{j + 1}.{a}" for j in range(5) for a in C.JOINTS]
    return names


def group_of(name: str) -> str:
    if name.startswith("eff."):
        return "eff_lag" + name.split(".")[1][3:]
    if name.startswith("blk"):
        return "sensor"
    return "vel" if name.startswith("qvel") else "orient"


# --------------------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------------------

def spectrum(dW: np.ndarray) -> tuple[float, float]:
    s2 = np.linalg.svd(dW, compute_uv=False) ** 2
    return float(s2[0] / s2.sum()), float(s2.sum() ** 2 / (s2 ** 2).sum())


def features(runs, W, W0) -> pd.DataFrame:
    rows = []
    for _, r in runs.iterrows():
        w, w0 = W[r.wandb_id], W0[r.wandb_id]
        f = {"wandb_id": r.wandb_id}
        for name in sorted(w):
            (k, b), (k0, b0) = w[name], w0[name]
            dk = k - k0
            f[f"{name}.relchange"] = np.linalg.norm(dk) / np.linalg.norm(k0)
            f[f"{name}.norm"] = np.linalg.norm(k)
            f[f"{name}.dW_top1"], f[f"{name}.dW_erank"] = spectrum(dk)
            f[f"{name}.bias_mean"] = b.mean()
            f[f"{name}.bias_std"] = b.std()
            f[f"{name}.dbias_norm"] = np.linalg.norm(b - b0)
        # head: 6 action means then 6 std pre-activations, labelled outputs
        k, b = w["actor4"]
        k0, _ = w0["actor4"]
        for j, jn in enumerate(C.JOINTS):
            f[f"head.mean_colnorm.{jn}"] = np.linalg.norm(k[:, j])
            f[f"head.mean_colratio.{jn}"] = np.linalg.norm(k[:, j]) / np.linalg.norm(k0[:, j])
            f[f"head.mean_bias.{jn}"] = b[j]
            f[f"head.std_colnorm.{jn}"] = np.linalg.norm(k[:, 6 + j])
            f[f"head.std_colratio.{jn}"] = (np.linalg.norm(k[:, 6 + j])
                                            / np.linalg.norm(k0[:, 6 + j]))
            f[f"head.std_bias.{jn}"] = b[6 + j]
            f[f"head.std_at_bias.{jn}"] = np.log(np.logaddexp(0, b[6 + j]) + MIN_STD)
        # first layer that sees the raw (delayed) observation, per labelled input row:
        # ratio of trained row norm to init row norm (scale-free across fan-ins).
        first = "predictor0" if r.forward_model else "actor0"
        k, _ = w[first]
        k0, _ = w0[first]
        names = input_names(r)
        assert len(names) == k.shape[0], (r.wandb_id, len(names), k.shape)
        ratio = np.linalg.norm(k, axis=1) / np.linalg.norm(k0, axis=1)
        dratio = np.linalg.norm(k - k0, axis=1) / np.linalg.norm(k0, axis=1)
        for n, v, dv in zip(names, ratio, dratio):
            f[f"in.ratio.{n}"] = v
            f[f"in.dratio.{n}"] = dv
        for g in sorted({group_of(n) for n in names}):
            idx = [i for i, n in enumerate(names) if group_of(n) == g]
            f[f"ingroup.ratio.{g}"] = ratio[idx].mean()
            f[f"ingroup.dratio.{g}"] = dratio[idx].mean()
        # FM decoder's input layer (sees the predicted obs, + nothing else)
        if r.forward_model:
            k, _ = w["actor0"]
            k0, _ = w0["actor0"]
            dn = list(C.OBS_NAMES) + (list(C.SENSOR_NAMES) if r.sensor else [])
            rr = np.linalg.norm(k, axis=1) / np.linalg.norm(k0, axis=1)
            for g in ("orient", "vel", "sensor"):
                idx = [i for i, n in enumerate(dn) if group_of(n) == g]
                if idx:
                    f[f"decoder_in.ratio.{g}"] = rr[idx].mean()
        rows.append(f)
    feats = pd.DataFrame(rows)
    meta = runs[["wandb_id", "condition", "arch", "seed", "seed_pair", "reward_480M",
                 *FACTORS]]
    return meta.merge(feats, on="wandb_id")


# --------------------------------------------------------------------------------------
# within-seed similarity
# --------------------------------------------------------------------------------------

def flat_delta(W, W0, wid, layers):
    return np.concatenate([(W[wid][l][0] - W0[wid][l][0]).ravel() for l in layers])


def similarity(runs, W, W0) -> pd.DataFrame:
    actor_l = [l for l in SHARED if l.startswith("actor")]
    critic_l = [l for l in SHARED if l.startswith("critic")]
    rep = {frozenset(p) for p in REPLICATES}
    rows = []
    by = runs.set_index("wandb_id")
    for seed, g in runs.groupby("seed"):
        ids = list(g.wandb_id)
        d = {w: {"actor": flat_delta(W, W0, w, actor_l),
                 "critic": flat_delta(W, W0, w, critic_l)} for w in ids}
        for a, b in itertools.combinations(ids, 2):
            same_init = {
                part: all(np.array_equal(W0[a][l][0], W0[b][l][0]) for l in ls)
                for part, ls in (("actor", actor_l), ("critic", critic_l))}
            row = dict(seed=seed, a=a, b=b, budget_a=by.at[a, "budget"],
                       budget_b=by.at[b, "budget"], cond_a=by.at[a, "condition"],
                       cond_b=by.at[b, "condition"],
                       seed_pair_a=by.at[a, "seed_pair"], seed_pair_b=by.at[b, "seed_pair"],
                       replicate=frozenset((a, b)) in rep,
                       init_shared_actor=same_init["actor"],
                       init_shared_critic=same_init["critic"])
            for f in FACTORS:
                row[f"diff.{f}"] = bool(by.at[a, f] != by.at[b, f])
            row["n_diff"] = sum(row[f"diff.{f}"] for f in FACTORS)
            for part in ("actor", "critic"):
                x, y = d[a][part], d[b][part]
                # unaligned hidden units (different init) make the cosine meaningless
                row[f"cos_{part}"] = (float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
                                      if same_init[part] else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------------------

def fit_logreg(X, y, lam=1.0):
    n, p = X.shape
    w0 = np.zeros(p + 1)
    sw = np.where(y == 1, 0.5 / max(y.mean(), 1e-9), 0.5 / max(1 - y.mean(), 1e-9))

    def loss(w):
        z = X @ w[1:] + w[0]
        l = np.logaddexp(0, z) - y * z
        g = (1 / (1 + np.exp(-z)) - y) * sw
        return (sw * l).sum() / n + 0.5 * lam * w[1:] @ w[1:] / n * 1.0, \
            np.concatenate([[g.sum() / n], X.T @ g / n + lam * w[1:] / n])

    return minimize(loss, w0, jac=True, method="L-BFGS-B").x


def loso_predict(X, y, groups, lam=1.0):
    pred = np.zeros(len(y))
    for gv in np.unique(groups):
        te = groups == gv
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            pred[te] = y[tr].mean()
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        w = fit_logreg((X[tr] - mu) / sd, y[tr], lam)
        pred[te] = (((X[te] - mu) / sd) @ w[1:] + w[0]) > 0
    return pred


def balanced_acc(y, p):
    return 0.5 * ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean())


def feature_sets(feats: pd.DataFrame) -> dict[str, list[str]]:
    shared = [c for c in feats.columns
              if any(c.startswith(l + ".") for l in SHARED) or c.startswith("head.")]
    shared = [c for c in shared if feats[c].notna().all()]
    proprio = [f"in.ratio.{n}" for n in C.OBS_NAMES]
    actor = [c for c in shared if not c.startswith("critic")]
    return {"shared_layers": shared, "actor_layers_only": actor,
            "shared+input_rows": shared + proprio}


def classify(feats: pd.DataFrame) -> pd.DataFrame:
    groups = feats.seed.to_numpy()
    rows = []
    for setname, cols in feature_sets(feats).items():
        X = feats[cols].to_numpy(float)
        for f in FACTORS:
            y = feats[f].to_numpy().astype(float)
            acc = balanced_acc(y, loso_predict(X, y, groups))
            null = np.array([balanced_acc(yp, loso_predict(X, yp, groups))
                             for yp in (RNG.permutation(y) for _ in range(N_PERM))])
            rows.append(dict(kind="factor", feature_set=setname, factor=f, n_features=len(cols),
                             n_pos=int(y.sum()), n=len(y), bal_acc=acc,
                             null_mean=null.mean(), null_q95=np.quantile(null, 0.95),
                             p_perm=(1 + (null >= acc).sum()) / (1 + N_PERM)))
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def residualize(X: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Remove each feature's linear dependence on reward (label-free, so no leakage)."""
    A = np.column_stack([np.ones_like(r), r])
    beta, *_ = np.linalg.lstsq(A, X, rcond=None)
    return X - A @ beta


#: (factor, subset name, mask) -- tests that the design's nesting can't explain.
SUBSETS = [
    ("privileged_critic", "MLP only", lambda f: ~f.forward_model),
    ("efference", "MLP only", lambda f: ~f.forward_model),
    ("efference", "FM only", lambda f: f.forward_model),
    ("servo", "MLP only", lambda f: ~f.forward_model),
    ("servo", "FM only", lambda f: f.forward_model),
    ("sensor", "MLP only", lambda f: ~f.forward_model),
    ("sensor", "FM only", lambda f: f.forward_model),
    ("forward_model", "eff+priv only", lambda f: f.efference & f.privileged_critic),
]


def classify_controls(feats: pd.DataFrame, n_perm: int = 200) -> pd.DataFrame:
    """Reward-residualised features, and each factor within a network subset."""
    rows = []
    cols = feature_sets(feats)["shared_layers"]
    r = feats.reward_480M.to_numpy(float)
    Xr = residualize(feats[cols].to_numpy(float), r)
    jobs = [("shared_layers, reward-residualised", f, "all", np.ones(len(feats), bool), Xr)
            for f in FACTORS]
    jobs += [("shared_layers", fac, name, m(feats).to_numpy(), feats[cols].to_numpy(float))
             for fac, name, m in SUBSETS]
    jobs += [("shared_layers, reward-residualised", fac, name, m(feats).to_numpy(), Xr)
             for fac, name, m in SUBSETS]
    acols = feature_sets(feats)["actor_layers_only"]
    Xa = residualize(feats[acols].to_numpy(float), r)
    jobs += [("actor_layers_only, reward-residualised", fac, name, m(feats).to_numpy(), Xa)
             for fac, name, m in SUBSETS]
    for setname, fac, subset, mask, X in jobs:
        y = feats.loc[mask, fac].to_numpy().astype(float)
        Xs, g = X[mask], feats.seed.to_numpy()[mask]
        acc = balanced_acc(y, loso_predict(Xs, y, g))
        null = np.array([balanced_acc(yp, loso_predict(Xs, yp, g))
                         for yp in (RNG.permutation(y) for _ in range(n_perm))])
        rows.append(dict(kind="factor_control", feature_set=setname, factor=fac,
                         subset=subset, n_features=Xs.shape[1], n_pos=int(y.sum()),
                         n=len(y), bal_acc=acc, null_mean=null.mean(),
                         null_q95=np.quantile(null, 0.95),
                         p_perm=(1 + (null >= acc).sum()) / (1 + n_perm)))
        print(rows[-1], flush=True)
    # how well do the same features predict reward itself (LOSO ridge R^2)?
    X = feats[cols].to_numpy(float)
    pred = np.zeros(len(r))
    for gv in np.unique(feats.seed):
        te = feats.seed.to_numpy() == gv
        mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-9
        Z = (X[~te] - mu) / sd
        beta = np.linalg.solve(Z.T @ Z + 10 * np.eye(Z.shape[1]), Z.T @ (r[~te] - r[~te].mean()))
        pred[te] = ((X[te] - mu) / sd) @ beta + r[~te].mean()
    r2 = 1 - ((r - pred) ** 2).sum() / ((r - r.mean()) ** 2).sum()
    rows.append(dict(kind="reward_from_weights", feature_set="shared_layers", bal_acc=np.nan,
                     r2_loso=r2))
    print(rows[-1])
    return pd.DataFrame(rows)


def single_feature(feats: pd.DataFrame, top=8) -> pd.DataFrame:
    """Leave-one-seed-out balanced accuracy of each feature alone, per factor."""
    groups = feats.seed.to_numpy()
    cols = feature_sets(feats)["shared+input_rows"]
    rows = []
    for f in FACTORS:
        y = feats[f].to_numpy().astype(float)
        accs = []
        for c in cols:
            x = feats[[c]].to_numpy(float)
            accs.append((balanced_acc(y, loso_predict(x, y, groups, lam=1e-3)), c,
                         float(np.corrcoef(x[:, 0], y)[0, 1])))
        for acc, c, r in sorted(accs, reverse=True)[:top]:
            rows.append(dict(kind="single_feature", factor=f, feature=c, bal_acc=acc,
                             corr_with_factor=r))
    return pd.DataFrame(rows)


def servo_pairs(feats: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    """Same-seed, same-arch, matched-reward torque/servo pairs: which features flip?"""
    pairs = []
    for tq, sv in SERVO_PAIRS:
        a = feats[feats.condition == tq]
        b = feats[feats.condition == sv]
        for seed in sorted(set(a.seed) & set(b.seed)):
            for _, ra in a[a.seed == seed].iterrows():
                for _, rb in b[b.seed == seed].iterrows():
                    pairs.append((tq, seed, ra, rb))
    cols = feature_sets(feats)["shared+input_rows"]
    cols += [c for c in feats.columns if c.startswith(("ingroup.", "decoder_in."))
             and feats.loc[feats.condition.isin([p for pp in SERVO_PAIRS for p in pp]), c]
             .notna().all()]
    D = np.array([[rb[c] - ra[c] for c in cols] for _, _, ra, rb in pairs])
    # one pair per (cell, seed) for the sign test: average over duplicate replicates
    keys = [(tq, s) for tq, s, _, _ in pairs]
    uk = sorted(set(keys))
    Du = np.array([D[[i for i, k in enumerate(keys) if k == u]].mean(0) for u in uk])
    sd = feats[cols].std().to_numpy() + 1e-12
    rows = []
    for j, c in enumerate(cols):
        frac = (Du[:, j] > 0).mean()
        rows.append(dict(kind="servo_pair_feature", feature=c, n_pairs=len(uk),
                         frac_servo_greater=frac, mean_diff_in_sd=(Du[:, j] / sd[j]).mean()))
    out = pd.DataFrame(rows)
    # leave-one-(cell,seed)-out pair discrimination with the mean standardized difference
    Z = Du / sd
    correct = []
    for i in range(len(uk)):
        w = np.delete(Z, i, 0).mean(0)
        correct.append(float(Z[i] @ w > 0))
    # pairwise ΔW similarity between the members of each servo pair vs replicate floor
    s = sim.copy()
    s["key"] = s.apply(lambda r: frozenset((r.cond_a, r.cond_b)), axis=1)
    pair_cos = s[s.key.isin([frozenset(p) for p in SERVO_PAIRS])]
    summary = pd.DataFrame([dict(
        kind="servo_pair_discrimination", n_pairs=len(uk),
        loo_pair_acc=float(np.mean(correct)),
        cos_actor_servo_pairs=pair_cos.cos_actor.mean(),
        cos_actor_replicates=s[s.replicate].cos_actor.mean())])
    return pd.concat([out, summary], ignore_index=True), pairs


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------

def figures(feats, sim, cls, single):
    import matplotlib.pyplot as plt

    from vnl_experiments.wandb_utils.style import apply_style
    apply_style()
    fig_dir = HERE / "figures"

    # 1. relative change per layer, by network/servo
    lay = ([f"predictor{i}" for i in range(5)] + [f"actor{i}" for i in range(5)]
           + [f"critic{i}" for i in range(6)])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    groups = [("MLP torque", (~feats.forward_model) & (~feats.servo), "C0", "o"),
              ("MLP servo", (~feats.forward_model) & feats.servo, "C0", "s"),
              ("FM torque", feats.forward_model & (~feats.servo), "C3", "o"),
              ("FM servo", feats.forward_model & feats.servo, "C3", "s")]
    for k, (lab, m, col, mk) in enumerate(groups):
        xs, ys, es = [], [], []
        for i, l in enumerate(lay):
            v = feats.loc[m, f"{l}.relchange"].dropna()
            if len(v):
                xs.append(i + (k - 1.5) * 0.15); ys.append(v.mean()); es.append(v.std())
        ax.errorbar(xs, ys, es, fmt=mk, color=col, ms=4, lw=0.8,
                    mfc="none" if "servo" in lab else col, label=f"{lab} (n={m.sum()})")
    ax.set_xticks(range(len(lay)))
    ax.set_xticklabels([l.replace("predictor", "pred").replace("actor", "act/dec")
                        .replace("critic", "crit") for l in lay], rotation=60, fontsize=7)
    ax.set_ylabel("||W − W0|| / ||W0||")
    ax.set_title("How far training moved each layer from its init (480 M)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(fig_dir / "weights_relchange.png", dpi=200); plt.close(fig)

    # 2. input reliance by group (first layer that sees the delayed obs), by factor
    ing = ["orient", "vel", "sensor"] + [f"eff_lag{j}" for j in range(1, 6)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    panels = [("servo", ("torque", "servo"), feats.forward_model, "servo: FM predictor"),
              ("servo", ("torque", "servo"), ~feats.forward_model, "servo: MLP actor"),
              ("forward_model", ("MLP actor", "FM predictor"), feats.efference,
               "MLP actor vs FM predictor (efference runs)")]
    for ax, (fac, labs, keep, title) in zip(axes, panels):
        for k, val in enumerate((False, True)):
            sub = feats[keep & (feats[fac] == val)]
            ys = [sub[f"ingroup.ratio.{g}"].mean() for g in ing]
            es = [sub[f"ingroup.ratio.{g}"].std() for g in ing]
            ax.errorbar(np.arange(len(ing)) + (k - .5) * .25, ys, es, fmt="o", ms=4,
                        color=f"C{k}", label=f"{labs[k]} (n={len(sub)})")
        ax.set_title(title, fontsize=9)
        ax.axhline(1, color="0.6", lw=0.8, ls="--")
        ax.set_xticks(range(len(ing))); ax.set_xticklabels(ing, rotation=50, fontsize=8)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("first-layer row norm / init row norm")
    fig.suptitle("Input reliance read from the first layer (mean ± sd over runs)", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "weights_input_reliance.png", dpi=200)
    plt.close(fig)

    # 3. similarity matrices per seed (actor dW)
    seeds = sorted(sim.seed.unique())
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.2 * len(seeds), 4.4))
    for ax, sd in zip(np.atleast_1d(axes), seeds):
        s = sim[sim.seed == sd]
        ids = sorted(set(s.a) | set(s.b),
                     key=lambda w: feats.set_index("wandb_id").at[w, "condition"])
        M = np.eye(len(ids))
        for _, r in s.iterrows():
            i, j = ids.index(r.a), ids.index(r.b)
            M[i, j] = M[j, i] = r.cos_actor
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        lab = [feats.set_index("wandb_id").at[w, "condition"] for w in ids]
        ax.set_xticks(range(len(ids))); ax.set_yticks(range(len(ids)))
        ax.set_xticklabels(lab, rotation=90, fontsize=5); ax.set_yticklabels(lab, fontsize=5)
        ax.set_title(f"init seed {sd}", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.6, label="cos(ΔW_a, ΔW_b), actor/decoder layers 1–4")
    fig.savefig(fig_dir / "weights_similarity_by_seed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4. what drives within-seed divergence: cos vs which factor differs (single-factor pairs)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    cats = [("replicate", sim.replicate)]
    for f in FACTORS:
        cats.append((f"only {f}", (sim.n_diff == 1) & sim[f"diff.{f}"]))
    cats.append(("2+ factors", sim.n_diff >= 2))
    for i, (lab, m) in enumerate(cats):
        v = sim.loc[m, "cos_actor"]
        ax.plot(np.full(len(v), i) + RNG.uniform(-.12, .12, len(v)), v, "o", ms=3,
                color="C0", alpha=.7)
        if len(v):
            ax.plot([i - .25, i + .25], [v.median()] * 2, color="k", lw=1.5)
        ax.text(i, -0.02, f"n={len(v)}", ha="center", va="top", fontsize=7)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels([c[0] for c in cats], rotation=30,
                                                        fontsize=8)
    ax.set_ylim(-0.1, 1); ax.set_ylabel("cos(ΔW) actor/decoder 1–4")
    ax.set_title("Same-seed pairs: how much do their weight updates agree?", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "weights_similarity_by_factor.png", dpi=200)
    plt.close(fig)

    # 5. classifier accuracy vs null: pooled (left) and within subsets, reward removed (right)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw=dict(width_ratios=[1, 1.5]))
    c = cls[cls.kind == "factor"]
    ax = axes[0]
    for k, (setname, g) in enumerate(c.groupby("feature_set", sort=False)):
        x = np.arange(len(g)) + (k - 1) * .27
        ax.bar(x, g.bal_acc, .25, color=f"C{k}", label=setname)
        ax.errorbar(x, g.null_mean, [g.null_mean * 0, g.null_q95 - g.null_mean], fmt="_",
                    color="k", ms=6, capsize=2)
    rr = cls[(cls.kind == "factor_control") & (cls.subset == "all")]
    ax.plot(np.arange(len(rr)), rr.bal_acc, "D", color="k", mfc="w", ms=6,
            label="shared_layers, reward-residualised")
    ax.axhline(0.5, color="0.6", lw=0.8, ls="--")
    ax.set_xticks(range(len(FACTORS))); ax.set_xticklabels(FACTORS, fontsize=8, rotation=20)
    ax.set_ylabel("balanced accuracy (leave-one-seed-out)"); ax.set_ylim(0, 1.08)
    ax.legend(fontsize=6, loc="lower left"); ax.set_title("all 62 runs", fontsize=9)
    ax = axes[1]
    cc = cls[(cls.kind == "factor_control") & (cls.subset != "all")]
    labels = list(dict.fromkeys(zip(cc.factor, cc.subset)))
    for k, (setname, g) in enumerate(cc.groupby("feature_set", sort=False)):
        g = g.set_index(["factor", "subset"]).loc[labels]
        x = np.arange(len(g)) + (k - 1) * .27
        ax.bar(x, g.bal_acc, .25, color=f"C{k + 3}", label=setname)
        ax.errorbar(x, g.null_mean, [g.null_mean * 0, g.null_q95 - g.null_mean], fmt="_",
                    color="k", ms=6, capsize=2)
    ax.axhline(0.5, color="0.6", lw=0.8, ls="--")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{f}\n({s})" for f, s in labels], fontsize=7, rotation=30)
    ax.set_ylim(0, 1.08); ax.legend(fontsize=6, loc="lower left")
    ax.set_title("within network subsets (black: permutation null mean to 95 %)", fontsize=9)
    fig.tight_layout(); fig.savefig(fig_dir / "weights_classify.png", dpi=200); plt.close(fig)

    # 6. head std biases / head mean column norms: servo vs torque
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, pre, yl in [(axes[0], "head.std_at_bias", "log std at bias (softplus(b)+1e-3)"),
                        (axes[1], "head.mean_colratio", "mean-head col norm / init")]:
        for k, (lab, m) in enumerate([("torque", ~feats.servo), ("servo", feats.servo)]):
            ys = [feats.loc[m, f"{pre}.{j}"].mean() for j in C.JOINTS]
            es = [feats.loc[m, f"{pre}.{j}"].std() for j in C.JOINTS]
            ax.errorbar(np.arange(6) + (k - .5) * .25, ys, es, fmt="o", ms=4, color=f"C{k}",
                        label=f"{lab} (n={m.sum()})")
        ax.set_xticks(range(6)); ax.set_xticklabels(C.JOINTS, fontsize=8); ax.set_ylabel(yl)
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "weights_head.png", dpi=200); plt.close(fig)


# --------------------------------------------------------------------------------------

def main():
    if "--figures-only" in sys.argv:
        feats = pd.read_csv(HERE / "weights_features.csv")
        sim = pd.read_csv(HERE / "weights_similarity.csv")
        cls = pd.read_csv(HERE / "weights_classify.csv")
        figures(feats, sim, cls, cls[cls.kind == "single_feature"])
        return
    runs = C.load_runs()
    W, W0 = load_all(runs)
    feats = features(runs, W, W0)
    feats.to_csv(HERE / "weights_features.csv", index=False, float_format="%.6g")
    sim = similarity(runs, W, W0)
    sim.to_csv(HERE / "weights_similarity.csv", index=False, float_format="%.5f")
    cls = pd.concat([classify(feats), classify_controls(feats)], ignore_index=True)
    single = single_feature(feats)
    sp, _ = servo_pairs(feats, sim)
    out = pd.concat([cls, single, sp], ignore_index=True)
    out.to_csv(HERE / "weights_classify.csv", index=False, float_format="%.4g")
    figures(feats, sim, cls, single)
    print(out[out.kind != "servo_pair_feature"].to_string())


if __name__ == "__main__":
    main()
