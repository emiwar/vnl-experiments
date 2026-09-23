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
    apply_style,
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

    if not CLIPS.exists():
        print("\nclips.csv not present: the per-behaviour figures need the per-clip eval\n"
              "artifacts. See this folder's report.md for the produce/pull commands.")

    write_figure_manifest(HERE, manifest)


if __name__ == "__main__":
    main()
