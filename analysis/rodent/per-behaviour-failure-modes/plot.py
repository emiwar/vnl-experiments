"""Figures for per-behaviour-failure-modes.

Reads only the committed CSVs in this folder -- no WandB, no artifact store, no network.

    ../.venv/bin/python analysis/rodent/per-behaviour-failure-modes/plot.py
    VNL_NO_FOOTER=1 ../.venv/bin/python analysis/rodent/per-behaviour-failure-modes/plot.py

Stage A figures (from ``data.csv``, which needs only the run index) decompose the cohort's
aggregate reward by termination reason and by reward term. The per-behaviour figures need
``clips.csv`` / ``behaviour.csv`` from the per-clip eval artifacts and are skipped with a
message until those exist, so this script runs at every stage.

Conventions followed here, from analysis/README.md §7: raw reward on the y-axis; small
multiples per dataset rather than a ratio, because a 30 s clip banks ~6x the reward of a
5 s one and dividing that out would assert a linearity the task does not have; the reward
source named in every axis label; and one colour per condition from
``style.CONDITION_STYLE``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vnl_experiments.wandb_utils.style import (
    SEED_LINE,
    apply_style,
    behaviour_color,
    behaviour_label,
    color_for,
    label_for,
    marker_for,
    provenance,
    reward_label,
    write_figure_manifest,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
DATA = HERE / "data.csv"
CLIPS = HERE / "clips.csv"
BEHAVIOUR = HERE / "behaviour.csv"

#: Datasets in the order they are panelled, with the `reward_label` source key for each.
DATASETS = ("train", "old_eval", "new_eval")

#: Conditions in reading order: the two ceilings, the subject and its floor, the
#: reward-matched control, the same ablation under torque, then the task-blind floor. Not
#: alphabetical -- this is the order the argument is made in, and it is the same in every
#: figure so the eye can compare panels.
ORDER = (
    "pos_intact",
    "torque_intact",
    "pos_noproprio_eff2",
    "torque_delay10",
    "pos_noproprio_eff0",
    "torque_noproprio_eff2",
    "pos_nointent",
    "torque_nointent",
)
#: Drawn, but visually set apart and excluded from any claim -- see extract.py's
#: `_aug11_torque_delay10` for why this run's provenance is weaker than the rest.
CROSS_CHECK = "torque_delay10_aug11"

#: Short forms for the categorical axis. The canonical `style.label_for` strings are too
#: long to tick eight of them without collision, and this is a *display* concern only --
#: the colour, which is what has to mean the same thing across folders, still comes from
#: `CONDITION_STYLE`. First line is the actuator, second the decoder input.
SHORT = {
    "pos_intact": "pos · all inputs",
    "torque_intact": "torque · all inputs",
    "pos_noproprio_eff2": "pos · no proprio, eff 2",
    "torque_delay10": "torque · proprio delay 10",
    "pos_noproprio_eff0": "pos · no proprio, eff 0",
    "torque_noproprio_eff2": "torque · no proprio, eff 2",
    "pos_nointent": "pos · no intention",
    "torque_nointent": "torque · no intention",
    CROSS_CHECK: "torque · delay 10 (aug-11)",
}

#: Failure terminations, in the order they are stacked. `survived` is the remainder and is
#: not a failure: the env sets `done` on clip-end truncation without setting any
#: `terminations/*` flag, so a "survived" episode is one that reached the end of the clip.
TERMINATIONS = ("root_too_far", "root_too_rotated", "pose_error", "nan_termination")
TERMINATION_STYLE = {
    "root_too_far": ("#c44e52", "Root too far (> 0.1 m)"),
    "root_too_rotated": ("#dd8452", r"Root too rotated (> 60$\degree$)"),
    "pose_error": ("#8172b3", "Pose error (joint L2 > 4.5)"),
    "nan_termination": ("#937860", "NaN in qpos"),
    "survived": ("#c7c7c7", "Reached end of clip"),
}

#: The reward terms with a non-zero weight in this env config, in the order they are
#: panelled: the four tracking terms, the near-free posture term, then the three costs.
#: `joints_vel` and `bodies_pos` are configured at weight 0 and are omitted from the
#: figure rather than drawn as empty panels; `data.csv` keeps their (zero) columns so the
#: omission is checkable.
ACTIVE_TERMS = ("root_pos", "root_quat", "joints", "end_eff", "torso_z_range",
                "control_cost", "control_diff_cost", "energy_cost")
TERM_LABEL = {
    "root_pos": "Root position",
    "root_quat": "Root orientation",
    "joints": "Joint angles",
    "end_eff": "End effectors",
    "torso_z_range": "Torso height in range",
    "control_cost": "Control cost",
    "control_diff_cost": "Control-difference cost",
    "energy_cost": "Energy cost",
}


def _present(df: pd.DataFrame) -> list[str]:
    """Conditions from ORDER that this CSV actually holds, in ORDER."""
    have = set(df["condition"])
    return [c for c in ORDER if c in have]


def _x_positions(conditions: list[str], cross: bool) -> dict[str, float]:
    """Categorical x positions, with a visible gap before the cross-check run."""
    pos = {c: float(i) for i, c in enumerate(conditions)}
    if cross:
        pos[CROSS_CHECK] = float(len(conditions)) + 0.6
    return pos


def _condition_axis(ax, pos: dict[str, float]) -> None:
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels([SHORT.get(c, label_for(c)) for c in pos],
                       rotation=40, ha="right", fontsize=7)
    ax.set_xlim(min(pos.values()) - 0.6, max(pos.values()) + 0.6)


def _scatter_runs(ax, sub: pd.DataFrame, column: str, pos: dict[str, float],
                  jitter: float = 0.13) -> None:
    """One marker per *run*, plus a mean bar.

    Every run is drawn rather than a mean with an error bar: with 1-3 runs per cell a
    standard error would imply a distribution that has not been sampled, and the n=1 cells
    would get an error bar of zero. The bar is the mean; the markers are the evidence.
    """
    rng = np.random.RandomState(0)
    for condition, group in sub.groupby("condition", observed=True):
        if condition not in pos:
            continue
        x = pos[condition]
        values = group[column].dropna().to_numpy()
        if not len(values):
            continue
        ax.hlines(values.mean(), x - 0.28, x + 0.28,
                  color=color_for(condition), lw=2.2, zorder=2)
        dx = rng.uniform(-jitter, jitter, len(values)) if len(values) > 1 else np.zeros(1)
        ax.plot(x + dx, values, marker=marker_for(condition), ms=5, ls="none",
                mfc="none" if condition.startswith("torque") else color_for(condition),
                mec=color_for(condition), mew=1.3, zorder=3)


def fig_aggregate_reward(df: pd.DataFrame) -> plt.Figure:
    """Episode reward per condition, one panel per dataset with its own y-axis.

    Small multiples rather than one panel or a ratio: `new_eval` clips are 30 s against
    5 s, so its reward is ~6x larger before anything about the policy enters, and the two
    are not on one axis. Reproduces the parent folder's headline from the run index alone,
    with the replicate spread visible.
    """
    conditions = _present(df)
    cross = CROSS_CHECK in set(df["condition"])
    pos = _x_positions(conditions, cross)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(13, 4.2))
    for ax, dataset in zip(axes, DATASETS):
        sub = df[(df.dataset == dataset) & df.reward.notna()]
        _scatter_runs(ax, sub, "reward", pos)
        ax.set_title(dataset, fontsize=9)
        ax.set_ylabel(reward_label(dataset))
        ax.set_ylim(bottom=0)
        _condition_axis(ax, pos)
    fig.suptitle("Episode reward by condition, per eval dataset "
                 "(bar = mean, marker = one run; open marker = torque)", fontsize=9)
    fig.tight_layout()
    return fig


def fig_reward_per_alive_step(df: pd.DataFrame) -> plt.Figure:
    """Reward per alive step, mean lifespan, and hazard rate.

    Episode reward is survival x tracking quality, and this figure separates them -- the
    one thing the aggregate cannot say. Reward per alive step is also the only reward axis
    comparable *across* datasets, and the hazard rate the only lifespan axis, because raw
    survival penalises a 30 s clip for having six times as many chances to fail
    (analysis/rodent/README.md).
    """
    conditions = _present(df)
    cross = CROSS_CHECK in set(df["condition"])
    pos = _x_positions(conditions, cross)
    panels = [
        ("reward_per_step", "Reward per alive step", None),
        ("lifespan_s", "Mean time alive (s)", None),
        ("hazard_rate", r"Failure hazard (s$^{-1}$)", "log"),
    ]
    fig, axes = plt.subplots(len(panels), len(DATASETS),
                             figsize=(13, 9), squeeze=False)
    for row, (column, ylabel, scale) in enumerate(panels):
        for col, dataset in enumerate(DATASETS):
            ax = axes[row][col]
            sub = df[(df.dataset == dataset) & df[column].notna()]
            _scatter_runs(ax, sub, column, pos)
            if scale:
                ax.set_yscale(scale)
            else:
                ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(dataset, fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel)
            _condition_axis(ax, pos) if row == len(panels) - 1 else ax.set_xticks([])
    fig.suptitle("Episode reward decomposed into tracking quality and survival", fontsize=9)
    fig.tight_layout()
    return fig


def fig_aggregate_failure_modes(df: pd.DataFrame) -> plt.Figure:
    """Stacked per-reason termination fractions, one panel per dataset.

    The direct answer to "compare reasons for termination". Read the *composition*, not
    only the height of the grey block: two conditions can share a survival fraction and
    fail for different reasons, which is what separates falling behind the target
    (`root_too_far`) from losing the body's orientation (`root_too_rotated`).
    """
    conditions = _present(df)
    cross = CROSS_CHECK in set(df["condition"])
    pos = _x_positions(conditions, cross)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(13, 4.6))
    for ax, dataset in zip(axes, DATASETS):
        sub = df[(df.dataset == dataset) & df.survived.notna()]
        means = sub.groupby("condition", observed=True)[
            [f"term_{t}" for t in TERMINATIONS] + ["survived"]].mean()
        for condition in pos:
            if condition not in means.index:
                continue
            bottom = 0.0
            for key in list(TERMINATIONS) + ["survived"]:
                column = "survived" if key == "survived" else f"term_{key}"
                height = float(means.loc[condition, column])
                colour, _ = TERMINATION_STYLE[key]
                ax.bar(pos[condition], height, bottom=bottom, width=0.62,
                       color=colour, edgecolor="white", lw=0.4)
                bottom += height
        ax.set_title(dataset, fontsize=9)
        ax.set_ylabel("Fraction of clips")
        ax.set_ylim(0, 1)
        _condition_axis(ax, pos)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TERMINATION_STYLE[k][0])
               for k in list(TERMINATIONS) + ["survived"]]
    labels = [TERMINATION_STYLE[k][1] for k in list(TERMINATIONS) + ["survived"]]
    axes[-1].legend(handles, labels, fontsize=6.5, frameon=False,
                    loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("How episodes ended, by condition "
                 "(mean over runs; grey = reached the end of the clip)", fontsize=9)
    fig.tight_layout()
    return fig


def fig_reward_terms(df: pd.DataFrame) -> plt.Figure:
    """Each reward term per alive step, one panel per term, for one dataset.

    Per alive step, not per episode: the stored terms are masked episode *totals*, so a
    policy that dies at 60 % of the clip banks 60 % of every term and the composition
    would mostly re-measure lifespan. Dividing by lifespan asks the different and intended
    question -- while it was alive, which parts of the reference did it track?

    `old_eval` only, because this figure is about composition rather than level and three
    copies of it would not add a third thing to look at.
    """
    dataset = "old_eval"
    conditions = _present(df)
    cross = CROSS_CHECK in set(df["condition"])
    pos = _x_positions(conditions, cross)
    sub = df[(df.dataset == dataset) & df.reward.notna()]
    ncol = 4
    nrow = int(np.ceil(len(ACTIVE_TERMS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.1 * nrow), squeeze=False)
    for i, term in enumerate(ACTIVE_TERMS):
        ax = axes[i // ncol][i % ncol]
        _scatter_runs(ax, sub, f"rtps_{term}", pos)
        ax.set_title(TERM_LABEL[term], fontsize=8.5)
        ax.axhline(0, color="0.8", lw=0.7, zorder=1)
        _condition_axis(ax, pos) if i // ncol == nrow - 1 else ax.set_xticks([])
        if i % ncol == 0:
            ax.set_ylabel("Reward per alive step")
    for j in range(len(ACTIVE_TERMS), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Where the reward comes from, per alive step "
                 f"({reward_label(dataset).lower()})", fontsize=9)
    fig.tight_layout()
    return fig


#: The task-blind floor for each actuator: same body, same actuator, proprioception
#: present, no imitation target. Used only by `fig_reward_above_floor`.
FLOOR = {"position": "pos_nointent", "torque": "torque_nointent"}
CEILING = {"position": "pos_intact", "torque": "torque_intact"}


def fig_reward_above_floor(df: pd.DataFrame) -> plt.Figure:
    """Raw per-step reward, and the share of it that is actually earned by imitating.

    Panel A is raw, as the convention requires. Panel B is the one derived axis in this
    folder, and it is justified rather than convenient: a large part of this reward
    function is collectable without knowing the imitation target at all --
    ``torso_z_range`` is ~1.0 per step for *every* condition including the task-blind
    one, and root position and orientation are ~70 % saturated there too. So the raw
    fraction-of-ceiling overstates how much imitation a policy is doing, by an amount
    that is itself a property of the reward function. Panel B rescales each actuator's
    per-step reward onto its **own** task-blind floor (0) and its **own** intact ceiling
    (1), which is the quantity the parent folder's headline was implicitly about.

    Per actuator, because the control costs differ ~6x between them: a floor borrowed
    across actuators would not bound the right thing.
    """
    dataset = "old_eval"
    conditions = _present(df)
    cross = CROSS_CHECK in set(df["condition"])
    pos = _x_positions(conditions, cross)
    sub = df[(df.dataset == dataset) & df.reward_per_step.notna()]
    means = sub.groupby("condition", observed=True)["reward_per_step"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    _scatter_runs(axes[0], sub, "reward_per_step", pos)
    axes[0].set_ylabel("Reward per alive step")
    axes[0].set_title("A. Raw", fontsize=9, loc="left")
    axes[0].set_ylim(bottom=0)

    for condition, x in pos.items():
        mode = "torque" if condition.startswith("torque") else "position"
        floor, ceiling = means.get(FLOOR[mode]), means.get(CEILING[mode])
        if floor is None or ceiling is None or condition not in means.index:
            continue
        values = sub.loc[sub.condition == condition, "reward_per_step"]
        scaled = (values - floor) / (ceiling - floor)
        axes[1].hlines(scaled.mean(), x - 0.28, x + 0.28,
                       color=color_for(condition), lw=2.2, zorder=2)
        axes[1].plot([x] * len(scaled), scaled, marker=marker_for(condition), ms=5,
                     ls="none", mec=color_for(condition), mew=1.3, zorder=3,
                     mfc="none" if condition.startswith("torque")
                     else color_for(condition))
    axes[1].axhline(0, color="0.5", lw=0.9, ls="--")
    axes[1].axhline(1, color="0.5", lw=0.9, ls="--")
    axes[1].set_ylabel("Share of the imitation-attributable reward")
    axes[1].set_title("B. Rescaled: 0 = same actuator's task-blind policy, "
                      "1 = same actuator intact", fontsize=9, loc="left")
    for ax in axes:
        _condition_axis(ax, pos)
    fig.suptitle(f"How much of the reward is actually earned by imitating "
                 f"({reward_label(dataset).lower()})", fontsize=9)
    fig.tight_layout()
    return fig


# ======================================================================================
# Stage B: per-behaviour figures (need clips.csv / behaviour.csv)
# ======================================================================================

#: Behaviours in a fixed order, sedentary -> dynamic, so the hypothesis under test would
#: read as a slope. It does not, which is the result.
BEHAVIOUR_ORDER = ("LGroom", "RGroom", "FaceGroom", "Rear", "Walk", "FastWalk")

#: The clip-wise reference every paired figure is taken against, and the null for the
#: interaction contrast. `torque_intact` is the null rather than zero: it is a *different
#: but equally competent* policy, so whatever interaction it shows against `pos_intact` is
#: the floor a "flat profile" claim has to be judged against.
PAIR_REF = "pos_intact"
PAIR_NULL = "torque_intact"

#: Conditions carrying the argument, in reading order. The others are in the CSVs.
BEHAVIOUR_SERIES = ("pos_intact", "torque_intact", "pos_noproprio_eff2",
                    "torque_delay10", "pos_noproprio_eff0", "torque_noproprio_eff2",
                    "pos_nointent", "torque_nointent")

N_BOOT = 4000


def _clip_reference(d: pd.DataFrame) -> pd.Series:
    """Per-clip reward of `PAIR_REF`, averaged over its runs."""
    return d[d.condition == PAIR_REF].groupby("clip_index")["episode_reward"].mean()


def _paired_delta(d: pd.DataFrame, condition: str, ref: pd.Series) -> pd.DataFrame:
    """Per-clip paired delta vs the reference, with each clip's behaviour attached."""
    sub = d[d.condition == condition]
    if sub.empty:
        return pd.DataFrame(columns=["delta", "behaviour", "coarse"])
    per = sub.groupby("clip_index")["episode_reward"].mean()
    labels = d.drop_duplicates("clip_index").set_index("clip_index")[
        ["behaviour", "coarse"]]
    out = pd.DataFrame({"delta": (per - ref).dropna()})
    return out.join(labels)


def _interaction(delta: pd.DataFrame, rng) -> tuple[float, float, float]:
    """Mean paired delta on grooming minus on the dynamic behaviours, + bootstrap CI.

    The single inferential claim in this folder, pre-specified in `extract.py`'s
    docstring. Bootstrapped over *clips* within the run pair, which is the unit the
    pairing licenses; the run-level component is shown as thin lines instead, because
    1-3 runs per cell cannot estimate it.
    """
    g = delta.loc[delta.coarse == "groom", "delta"].to_numpy()
    l = delta.loc[delta.coarse.isin(["locomote", "rear"]), "delta"].to_numpy()
    if not len(g) or not len(l):
        return float("nan"), float("nan"), float("nan")
    boot = np.array([rng.choice(g, len(g), True).mean() - rng.choice(l, len(l), True).mean()
                     for _ in range(N_BOOT)])
    return g.mean() - l.mean(), *np.percentile(boot, [2.5, 97.5])


def fig_reward_by_behaviour(clips: pd.DataFrame) -> plt.Figure:
    """Mean per-clip reward per behaviour, one series per condition. **The primary figure.**

    Raw reward, no ratio: every clip is 250 mocap frames, so the six bins are directly
    comparable on one axis and no normalisation is needed to read them together. Error
    bars are the standard error over the *clips* in the bin for that condition; thin lines
    are individual runs, so an n=1 cell shows no spread rather than a fake one.

    The hypothesis this folder was built to test predicts that `pos_noproprio_eff2` tracks
    the ceiling on the three groom bins and falls away on Rear/Walk/FastWalk -- i.e. a
    downward slope left to right. Read whether it does.
    """
    datasets = [d for d in ("old_eval", "train") if d in set(clips.dataset)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.4 * len(datasets), 4.8),
                             squeeze=False)
    x = np.arange(len(BEHAVIOUR_ORDER))
    for ax, dataset in zip(axes[0], datasets):
        d = clips[clips.dataset == dataset]
        for condition in BEHAVIOUR_SERIES:
            sub = d[d.condition == condition]
            if sub.empty:
                continue
            per_run = sub.groupby(["wandb_id", "behaviour"], observed=True)[
                "episode_reward"].mean().unstack().reindex(columns=BEHAVIOUR_ORDER)
            for _, row in per_run.iterrows():
                ax.plot(x, row.to_numpy(), color=color_for(condition), **SEED_LINE)
            mean = per_run.mean(axis=0).to_numpy()
            sem = sub.groupby("behaviour", observed=True)["episode_reward"].sem(
                ).reindex(BEHAVIOUR_ORDER).to_numpy()
            ax.errorbar(x, mean, yerr=sem, color=color_for(condition),
                        marker=marker_for(condition), ms=5, lw=2.0, capsize=2,
                        mfc="none" if condition.startswith("torque") else None,
                        label=SHORT.get(condition, condition).replace("\n", " "))
        n = d.drop_duplicates("clip_index").behaviour.value_counts()
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}\n(n={int(n.get(b, 0))})" for b in BEHAVIOUR_ORDER],
                           fontsize=7.5)
        ax.set_xlabel("sedentary " + r"$\longrightarrow$" + " dynamic", fontsize=8)
        ax.set_ylabel(reward_label(dataset))
        ax.set_title(dataset, fontsize=9)
        ax.set_ylim(bottom=0)
    axes[0][0].legend(fontsize=6.5, frameon=False, ncol=2, loc="lower left")
    fig.suptitle("Mean per-clip reward by behaviour (bars = SEM over clips; "
                 "thin lines = individual runs)", fontsize=9)
    fig.tight_layout()
    return fig


def fig_paired_deficit(clips: pd.DataFrame) -> plt.Figure:
    """Paired per-clip deficit vs the clip-wise position-intact reference, and its slope.

    Panel A is the paired delta per behaviour: pairing removes clip difficulty, which is
    the instrument here -- the bins differ in intrinsic hardness (a task-blind policy
    scores 63 % of the ceiling on grooming and 18 % on fast walking), so an unpaired
    comparison across bins mostly measures the clips. The raw levels are in
    `reward_by_behaviour.png` alongside.

    Panel B is the folder's single inferential claim: the grooming deficit minus the
    dynamic-behaviour deficit, with a bootstrap CI over clips. The grey band is
    `torque_intact` measured the same way -- a different but equally competent policy --
    so it is what "no behaviour selectivity" actually looks like, rather than zero.
    """
    dataset = "old_eval"
    d = clips[clips.dataset == dataset]
    ref = _clip_reference(d)
    rng = np.random.RandomState(0)
    x = np.arange(len(BEHAVIOUR_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    series = [c for c in BEHAVIOUR_SERIES if c != PAIR_REF]
    for condition in series:
        delta = _paired_delta(d, condition, ref)
        if delta.empty:
            continue
        m = delta.groupby("behaviour", observed=True)["delta"].mean().reindex(
            BEHAVIOUR_ORDER)
        e = delta.groupby("behaviour", observed=True)["delta"].sem().reindex(
            BEHAVIOUR_ORDER)
        axes[0].errorbar(x, m.to_numpy(), yerr=e.to_numpy(), color=color_for(condition),
                         marker=marker_for(condition), ms=5, lw=2.0, capsize=2,
                         mfc="none" if condition.startswith("torque") else None,
                         label=SHORT.get(condition, condition).replace("\n", " "))
    axes[0].axhline(0, color="0.4", lw=1.0, ls="--")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(BEHAVIOUR_ORDER, fontsize=7.5)
    axes[0].set_xlabel("sedentary " + r"$\longrightarrow$" + " dynamic", fontsize=8)
    axes[0].set_ylabel(f"Paired $\\Delta$ reward vs position-intact\n({dataset}, same clip)")
    axes[0].set_title("A. Deficit per behaviour", fontsize=9, loc="left")
    axes[0].legend(fontsize=6.5, frameon=False, ncol=2, loc="lower left")

    null, null_lo, null_hi = _interaction(_paired_delta(d, PAIR_NULL, ref), rng)
    axes[1].axhspan(null_lo, null_hi, color="0.85", zorder=0)
    axes[1].axhline(0, color="0.4", lw=1.0, ls="--")
    for i, condition in enumerate(series):
        delta = _paired_delta(d, condition, ref)
        if delta.empty:
            continue
        obs, lo, hi = _interaction(delta, rng)
        axes[1].errorbar([i], [obs], yerr=[[obs - lo], [hi - obs]],
                         color=color_for(condition), marker=marker_for(condition),
                         ms=6, capsize=3, lw=1.6,
                         mfc="none" if condition.startswith("torque") else None)
    axes[1].set_xticks(range(len(series)))
    axes[1].set_xticklabels([SHORT.get(c, c).replace("\n", " ") for c in series],
                            rotation=40, ha="right", fontsize=7)
    axes[1].set_ylabel(r"Grooming $\Delta$ $-$ dynamic $\Delta$")
    axes[1].set_title(f"B. Behaviour selectivity (grey = {PAIR_NULL}, the no-selectivity "
                      f"null)", fontsize=9, loc="left")
    fig.suptitle("Is the deficit behaviour-selective? Paired per-clip comparison, "
                 "bootstrap CI over clips", fontsize=9)
    fig.tight_layout()
    return fig


#: A clip counts as "tracked" when it scores at least this fraction of what
#: `PAIR_REF` scored on the *same* clip. The threshold is not tuned: the per-clip ratio
#: distribution is strongly bimodal (see `fig_clip_outcomes`), with a dense mode at
#: 0.90-1.05 and a sparse tail below 0.5 and almost nothing between, so anything from
#: ~0.7 to ~0.85 partitions it identically. 0.9 is quoted because it is the edge of the
#: upper mode.
TRACKED_THRESHOLD = 0.9

#: Minimum tracked clips before "how well it tracks when it tracks" is drawn at all.
MIN_TRACKED_FOR_QUALITY = 5


def _ratio_table(d: pd.DataFrame, conditions) -> dict:
    """Per-clip reward as a fraction of `PAIR_REF` on the same clip, per condition."""
    ref = _clip_reference(d)
    labels = d.drop_duplicates("clip_index").set_index("clip_index")["behaviour"]
    out = {}
    for condition in conditions:
        sub = d[d.condition == condition]
        if sub.empty:
            continue
        per = sub.groupby("clip_index")["episode_reward"].mean()
        ratio = (per / ref).dropna()
        out[condition] = pd.DataFrame({"ratio": ratio,
                                       "behaviour": labels.reindex(ratio.index)})
    return out


def fig_clip_outcomes(clips: pd.DataFrame) -> plt.Figure:
    """Every clip's reward as a fraction of the same clip under position-intact.

    The distribution the per-behaviour means were taken over, and the reason those means
    are interpretable at all: it is **bimodal**. A clip either lands in a dense mode just
    below 1.0 -- tracked about as well as an intact policy tracks it -- or it collapses
    below 0.5. Very little sits between. So a mean per behaviour is close to
    ``P(tracked) x 0.95``, and the question "which behaviours does it fail at" is really a
    question about *how often* it fails, not about how well it tracks when it does not.

    Plotted as a ratio rather than raw reward, which the folder otherwise avoids: here the
    paired denominator is the point, and the raw levels are in `reward_by_behaviour.png`.
    A ratio is safe in this one place because the denominator barely varies -- the intact
    policy scores 2007-2256 on 95 % of clips.
    """
    dataset = "old_eval"
    d = clips[clips.dataset == dataset]
    shown = [c for c in ("pos_noproprio_eff2", "torque_delay10", "pos_noproprio_eff0",
                         "pos_nointent") if c in set(d.condition)]
    tables = _ratio_table(d, shown)
    rng = np.random.RandomState(0)
    fig, axes = plt.subplots(1, len(shown), figsize=(3.4 * len(shown), 4.4),
                             sharey=True, squeeze=False)
    x = np.arange(len(BEHAVIOUR_ORDER))
    for ax, condition in zip(axes[0], shown):
        t = tables[condition]
        for i, behaviour in enumerate(BEHAVIOUR_ORDER):
            vals = t.loc[t.behaviour == behaviour, "ratio"].to_numpy()
            if not len(vals):
                continue
            ax.plot(i + rng.uniform(-0.26, 0.26, len(vals)), vals, ls="none",
                    marker="o", ms=3, alpha=0.6, mec="none",
                    color=color_for(condition))
            ax.hlines(np.median(vals), i - 0.34, i + 0.34, color="0.15", lw=1.6,
                      zorder=3)
        ax.axhline(1.0, color="0.5", lw=0.9, ls="--")
        ax.axhline(TRACKED_THRESHOLD, color="0.7", lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(BEHAVIOUR_ORDER, rotation=45, ha="right", fontsize=6.5)
        ax.set_title(SHORT.get(condition, condition).replace("\n", " "), fontsize=8.5)
    axes[0][0].set_ylabel("Per-clip reward / position-intact on the same clip")
    fig.suptitle(f"Per-clip outcomes are bimodal (black bar = median; dotted = the "
                 f"{TRACKED_THRESHOLD:g} 'tracked' threshold; {dataset})", fontsize=9)
    fig.tight_layout()
    return fig


def fig_tracked_rate(clips: pd.DataFrame) -> plt.Figure:
    """Decompose each per-behaviour mean into *how often* it tracks and *how well*.

    Given the bimodality in `fig_clip_outcomes`, the mean is close to the product of
    these two panels. Panel A is the interesting one: it is where behaviour selectivity
    lives. Panel B is flat for every condition, which is the claim that when these
    policies track at all, they track about as well as an intact policy does -- and that
    what an ablation costs is the *rate*, not the quality.
    """
    dataset = "old_eval"
    d = clips[clips.dataset == dataset]
    shown = [c for c in BEHAVIOUR_SERIES if c in set(d.condition) and c != PAIR_REF]
    tables = _ratio_table(d, shown)
    x = np.arange(len(BEHAVIOUR_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)
    for condition in shown:
        t = tables[condition]
        rate, quality = [], []
        for behaviour in BEHAVIOUR_ORDER:
            vals = t.loc[t.behaviour == behaviour, "ratio"].to_numpy()
            tracked = vals[vals >= TRACKED_THRESHOLD]
            rate.append(np.mean(vals >= TRACKED_THRESHOLD) if len(vals) else np.nan)
            # Panel B is undefined where the policy almost never tracks: the task-blind
            # runs track 0 of 35 Rear clips, and a mean over the one or two that scrape
            # the threshold is noise that reads as a real excursion. Below this many
            # tracked clips the point is dropped rather than drawn.
            quality.append(tracked.mean() if len(tracked) >= MIN_TRACKED_FOR_QUALITY
                           else np.nan)
        style = dict(color=color_for(condition), marker=marker_for(condition), ms=5,
                     lw=2.0, mfc="none" if condition.startswith("torque") else None)
        axes[0].plot(x, rate, label=SHORT.get(condition, condition).replace("\n", " "),
                     **style)
        axes[1].plot(x, quality, **style)
    axes[0].set_ylabel(f"Fraction of clips tracked ($\\geq$ {TRACKED_THRESHOLD:g} of intact)")
    axes[0].set_title("A. How often it tracks", fontsize=9, loc="left")
    axes[0].set_ylim(0, 1.05)
    axes[1].axhline(1.0, color="0.5", lw=0.9, ls="--")
    axes[1].set_ylabel("Mean ratio on the clips it tracked")
    axes[1].set_title("B. How well, when it does", fontsize=9, loc="left")
    # Wide enough for `torque_intact`, which exceeds 1.0 because the denominator is the
    # *position*-intact policy and torque tracks slightly better.
    axes[1].set_ylim(0.85, 1.12)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(BEHAVIOUR_ORDER, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("sedentary " + r"$\longrightarrow$" + " dynamic", fontsize=8)
    fig.suptitle(f"An ablation costs the *rate* at which the policy tracks, not the "
                 f"quality when it does ({dataset})", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    return fig


def fig_failure_modes_by_behaviour(clips: pd.DataFrame) -> plt.Figure:
    """How episodes ended, per behaviour, one panel per condition.

    Built from ``term_reason``, which is an exact partition of the episodes -- unlike the
    aggregate per-reason rates in figure 3, which count each reason independently and so
    overflow slightly where two fire at once. The ``root_too_far+root_too_rotated`` band
    is that overlap, 0.21 % of episodes here.
    """
    dataset = "old_eval"
    d = clips[clips.dataset == dataset]
    shown = [c for c in BEHAVIOUR_SERIES if c in set(d.condition)]
    order = ["survived", "root_too_far", "root_too_rotated",
             "root_too_far+root_too_rotated", "pose_error", "nan_termination"]
    colour = {"survived": "#c7c7c7", "root_too_far": "#c44e52",
              "root_too_rotated": "#dd8452",
              "root_too_far+root_too_rotated": "#8c4a4f", "pose_error": "#8172b3",
              "nan_termination": "#937860"}
    ncol = 4
    nrow = int(np.ceil(len(shown) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.4 * nrow), squeeze=False,
                             sharey=True)
    x = np.arange(len(BEHAVIOUR_ORDER))
    present = [r for r in order if r in set(d.term_reason)]
    for i, condition in enumerate(shown):
        ax = axes[i // ncol][i % ncol]
        sub = d[d.condition == condition]
        frac = (sub.groupby(["behaviour", "term_reason"], observed=True).size()
                / sub.groupby("behaviour", observed=True).size()).unstack(
                    fill_value=0.0).reindex(BEHAVIOUR_ORDER).reindex(
                        columns=present, fill_value=0.0)
        bottom = np.zeros(len(BEHAVIOUR_ORDER))
        for reason in present:
            h = frac[reason].to_numpy()
            ax.bar(x, h, bottom=bottom, width=0.7, color=colour[reason],
                   edgecolor="white", lw=0.3)
            bottom += h
        ax.set_title(SHORT.get(condition, condition).replace("\n", " "), fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(BEHAVIOUR_ORDER, rotation=45, ha="right", fontsize=6.5)
        ax.set_ylim(0, 1)
        if i % ncol == 0:
            ax.set_ylabel("Fraction of clips")
    for j in range(len(shown), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour[r]) for r in present]
    axes[0][-1].legend(handles, [r.replace("+", " +\n") for r in present], fontsize=6,
                       frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(f"How episodes ended, by behaviour ({dataset}; exact partition)",
                 fontsize=9)
    fig.tight_layout()
    return fig


def main() -> None:
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)

    builders = [
        ("aggregate_reward", fig_aggregate_reward, (DATA,)),
        ("reward_per_alive_step", fig_reward_per_alive_step, (DATA,)),
        ("aggregate_failure_modes", fig_aggregate_failure_modes, (DATA,)),
        ("reward_terms", fig_reward_terms, (DATA,)),
        ("reward_above_floor", fig_reward_above_floor, (DATA,)),
    ]

    manifest = {}
    for name, builder, inputs in builders:
        fig = builder(df)
        manifest[f"{name}.png"] = provenance(fig, HERE, *inputs)
        fig.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}.png")

    if CLIPS.exists():
        clips = pd.read_csv(CLIPS)
        for name, builder in [
            ("reward_by_behaviour", fig_reward_by_behaviour),
            ("paired_deficit_by_behaviour", fig_paired_deficit),
            ("clip_outcomes", fig_clip_outcomes),
            ("tracked_rate", fig_tracked_rate),
            ("failure_modes_by_behaviour", fig_failure_modes_by_behaviour),
        ]:
            fig = builder(clips)
            manifest[f"{name}.png"] = provenance(fig, HERE, CLIPS)
            fig.savefig(FIGURES / f"{name}.png", dpi=200)
            plt.close(fig)
            print(f"wrote figures/{name}.png")
    else:
        print("\nclips.csv not present: the per-behaviour figures need the per-clip eval\n"
              "artifacts. See this folder's report.md for the produce/pull commands.")

    if not (HERE / "behaviour_frames.csv").exists():
        print("behaviour_frames.csv not present: the per-frame behaviour figures need the\n"
              "`trace` artifacts. See report.md for the produce/pull commands.")

    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
