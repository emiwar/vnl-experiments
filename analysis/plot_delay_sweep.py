"""Fetch and plot the proprioceptive delay-sweep results from WandB.

Run as::

    python analysis/plot_delay_sweep.py

Two lines (both AbsoluteImitation, TrainEvalSplit tag, same git/hyperparams):
  - With efference copy: efference_length == delay_k
  - No efference copy:   efference_length == 0, delay_k > 0
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

WANDB_PROJECT = "emiwar-team/nnx-ppo-rodent-delays"
OUTPUT = "analysis/delay_sweep_plot.png"
CTRL_DT_MS = 10  # 1 step = 10 ms (ctrl_dt = 0.01 s)

api = wandb.Api(timeout=60)
print("Fetching runs…")
all_runs = list(api.runs(WANDB_PROJECT, per_page=300, order="-created_at"))
finished = [r for r in all_runs if r.state == "finished"]
print(f"  {len(all_runs)} total, {len(finished)} finished")

# ── Group 1: with efference copy – TrainEvalSplit, eff_length == delay_k ──────
primary = [
    r for r in finished
    if "TrainEvalSplit" in r.tags
    and r.config.get("efference_length") == r.config.get("delay_k")
]

# ── Group 2: no efference copy – TrainEvalSplit, eff_length == 0, delay > 0 ───
# (delay=0 eff=0 is identical to the baseline in group 1, so exclude it)
no_eff = [
    r for r in finished
    if "TrainEvalSplit" in r.tags
    and r.config.get("efference_length") == 0
    and r.config.get("delay_k", 0) > 0
]

# ── Comparability report ───────────────────────────────────────────────────────
def config_row(r):
    c = r.config
    net = c.get("net_params", {})
    ppo_cfg = c.get("config", {}).get("ppo", {}) if isinstance(c.get("config"), dict) else {}
    git = "?"
    if hasattr(r, "metadata") and r.metadata:
        git = r.metadata.get("git", {}).get("commit", "?")[:8]
    return {
        "run": r.name[:45],
        "env": c.get("env"),
        "git": git,
        "delay_k": c.get("delay_k"),
        "efference": c.get("efference_length"),
        "latent_size": net.get("latent_size"),
        "kl_weight": net.get("kl_weight"),
        "body_target_frame": net.get("body_target_frame"),
        "n_envs": ppo_cfg.get("n_envs"),
        "total_steps": ppo_cfg.get("total_steps"),
        "reward": r.summary.get("episode_reward/mean"),
    }

def check_group(label, runs_list):
    rows = [config_row(r) for r in runs_list]
    df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print(f"Group: {label}  (n={len(runs_list)})")
    print(f"{'='*60}")
    for col in ["env", "git", "latent_size", "kl_weight", "body_target_frame", "n_envs", "total_steps"]:
        vals = df[col].unique()
        flag = "  *** VARIES ***" if len(vals) > 1 else ""
        print(f"  {col:20s}: {vals}{flag}")
    print(f"  delays covered: {sorted(df['delay_k'].dropna().unique().tolist())}")
    return df

df_primary_full = check_group("With efference copy (AbsoluteImitation, eff=delay)", primary)
df_no_eff_full  = check_group("No efference copy (AbsoluteImitation, eff=0)", no_eff)

# ── Dedup by delay_k (keep highest reward when multiple runs share a delay) ────
def dedup(df):
    df = df.dropna(subset=["delay_k", "reward"])
    df = df.sort_values("reward", ascending=False).drop_duplicates("delay_k")
    return df.sort_values("delay_k").reset_index(drop=True)

df_with    = dedup(df_primary_full)
df_without = dedup(df_no_eff_full)

print("\n\n── Final data for plot ──────────────────────────────────────────────")
print("With efference copy:")
print(df_with[["delay_k", "efference", "reward"]].to_string(index=False))
print("\nNo efference copy (AbsoluteImitation):")
print(df_without[["delay_k", "efference", "reward"]].to_string(index=False))

# ── Plot ───────────────────────────────────────────────────────────────────────
sns.set_theme(style="ticks")

max_delay = max(
    df_with["delay_k"].max() if len(df_with) else 0,
    df_without["delay_k"].max() if len(df_without) else 0,
)
xlim = (0, max_delay * 1.05)

fig, ax1 = plt.subplots(figsize=(5, 3.5))
ax2 = ax1.twiny()  # top x-axis: ms

if len(df_with) > 0:
    ax1.plot(df_with["delay_k"], df_with["reward"],
             color="C1", marker="o", ms=4, lw=1.8,
             label="With efference copy")
if len(df_without) > 0:
    ax1.plot(df_without["delay_k"], df_without["reward"],
             color="C0", marker="s", ms=4, lw=1.8,
             label="No efference copy")

ax1.set_xlim(xlim)
ax1.set_ylim(bottom=0)
ax1.set_xlabel("Observation delay (steps)")
ax1.set_ylabel("Mean episode reward")

ticks = ax1.get_xticks()
ticks = ticks[(ticks >= 0) & (ticks <= max_delay * 1.1)]
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(ticks)
ax2.set_xticklabels([f"{int(t * CTRL_DT_MS)}" for t in ticks])
ax2.set_xlabel("Observation delay (ms)")

ax1.legend(fontsize=8, frameon=False, loc="upper right")

sns.despine(ax=ax1)
sns.despine(ax=ax2, top=False, right=True, left=True, bottom=True)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {OUTPUT}")
