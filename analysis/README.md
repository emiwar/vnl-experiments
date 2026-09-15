# Analysis pipeline

Reproducible analyses of the WandB runs. One folder per scientific question. The goals:

* a figure rebuilds from **exactly** the runs it was made from, by default, forever;
* adding new runs to a figure is **one flag and a visible diff**, never a silent change;
* every expensive thing computed from a run is **kept, named, and traceable** to the code
  that made it, and reused across questions rather than recomputed;
* every comparison is **explicitly verified to be fair** before a conclusion is drawn.

Three layers support this. Read §1–§3 once; §4 onwards is the day-to-day workflow.

```
   WandB  ──sync──▶  analysis/_runs/*.jsonl        (§1 run index — configs + summaries)
                            │
   checkpoints ─produce─▶  $VNL_ARTIFACTS/         (§2 artifact store — evals, curves,
                            │                          activations, videos)
                            ▼
                     analysis/<track>/<question>/  (§3 the question folder)
                       extract.py → data.csv → plot.py → figures/ → report.md
```

### Two tracks

Question folders live under the experiment track they belong to. The tracks are separate
WandB projects with different reward scales, different eval vocabularies and different
control timesteps, so a figure pooling them is almost always a mistake — the split is
there to make that mistake visible rather than convenient.

| track | subject | project | read first |
|---|---|---|---|
| [`rodent/`](rodent/) | mocap imitation by the simulated rodent | `nnx-ppo-rodent-delays` | [`rodent/README.md`](rodent/README.md) |
| [`dm_control_suite/`](dm_control_suite/) | dm_control tasks with delays | `nnx-ppo-delays` | [`dm_control_suite/README.md`](dm_control_suite/README.md) |

This file is the machinery, and applies to both. Each track's README carries its project,
its reward vocabulary and its own trap list; **read the track README before starting a
question in it**, not only this one.

---

## 1. The run index

`analysis/_runs/<project>.jsonl` is a committed mirror of every run's metadata: the full
`config`, the scalar `summary`, and `state` / `created_at` / `tags` / `git_commit` /
`gpu` / `host`. One JSON object per line, sorted by `wandb_id`, so `git diff` after a
sync shows exactly which runs entered the project.

```bash
python -m vnl_experiments.wandb_utils.index sync   # incremental; only new/changed runs
python -m vnl_experiments.wandb_utils.index info
```

```python
from vnl_experiments.wandb_utils import index

df = index.load()                    # ~0.08 s for 466 runs, no network
runs = index.select(df, tags="TrainEvalSplit", state="finished",
                    **{"env_params.walker_xml_path": NEW_XML})
```

Columns are **dotted**: `env_params.body_target_frame`, `config.ppo.n_envs`,
`summary.eval/episode_reward/mean`. List-valued config entries are stored as their JSON
string (`"[512, 512, 512, 512]"`) so they stay hashable and comparable. `index.select`
raises on an unknown column rather than returning an empty cohort.

**Why:** a full WandB fetch is ~70 s to list the project plus ~1 s per run for
config/summary/metadata. Seventeen analyses each paying that, repeatedly, was the single
biggest cost in the pipeline. Sync is deliberate and occasional; reading is free.

## 2. The artifact store

Anything expensive computed *from* a run — an offline evaluation, a training history,
recorded activations, a rollout video — is an **artifact**, keyed by
`(kind, wandb_id, spec_id)` under `$VNL_ARTIFACTS` (default `<repo>/artifacts/`,
gitignored):

```
artifacts/eval/<wandb_id>/<spec_id>.json
                          <spec_id>.meta.json     ← how it was made
          history/<wandb_id>/<spec_id>.csv.gz
          timing/<wandb_id>/<spec_id>.csv.gz
          activations/<wandb_id>/<spec_id>.h5
          video/<wandb_id>/<spec_id>.mp4  (+ .h5, .stats.json)
```

Every artifact carries a **sidecar** `.meta.json` with its full spec, the producer
(module, version, git commit, host, GPU), a checksum, and a `resolved` block of facts
discovered while producing (notably the `checkpoint_step` actually restored). The
sidecar, not any central file, is the source of truth — so `rsync` a directory between
the cluster and the laptop and the provenance travels with the bytes.

`analysis/_artifacts/manifest.jsonl` is a **committed index** of the store, rebuildable
at any time with `reindex`. It lets an analysis's requirements be checked, and the
history of what was computed be read, without the (multi-gigabyte) store present.

```bash
python -m vnl_experiments.artifacts ls
python -m vnl_experiments.artifacts plan   --kind eval --runs analysis/<q>/runs.csv --out todo.txt
python -m vnl_experiments.artifacts ensure --kind history --runs analysis/<q>/runs.csv
python -m vnl_experiments.artifacts pull   --kind eval --runs analysis/<q>/runs.csv
python -m vnl_experiments.artifacts verify
```

### The five kinds

| kind | what | cost | where it can run |
|---|---|---|---|
| `history` | sampled training curves + throughput, per run | seconds | anywhere (WandB) |
| `timing` | per-iteration wall clock + the three throughput gauges | seconds | anywhere (WandB) |
| `eval` | offline re-evaluation on train / `old_eval` / `new_eval` | minutes, GPU | needs a checkpoint |
| `activations` | per-layer unit activations on one dataset | minutes, 1–2 GB each | needs a checkpoint |
| `video` | rendered rollout mp4 + qpos h5 + stats | minutes, GPU | needs a checkpoint |

`history` and `timing` overlap in what they fetch but not in what they are for, and they
are separate kinds for a mechanical reason, not a stylistic one: WandB's sampled history
drops any row where a requested key is missing, so one `run.history(keys=[...])` call for
metrics logged at different cadences returns only their *intersection*. `history` asks for
the reward series and gets it aligned; `timing` needs the train / eval / video gauges,
which fire every iteration, every ~2.4 iterations and every ~24 iterations, so it fetches
them separately and joins on `_step`. Read a `timing` artifact through
`wandb_utils/timing.py`, which reconstructs a per-run budget (training / eval / video /
checkpoint / everything else) from it and the run's config — see
[`dm_control_suite/where-the-wall-clock-goes/`](dm_control_suite/where-the-wall-clock-goes/).

Activations and videos are the artifacts most worth reusing across questions: they are
question-independent and expensive, so record once, `pull` selectively, and let several
analyses `REQUIRE` the same `spec_id`.

### `spec_id`, and why it changes

`spec_id` is a readable prefix plus an 8-char hash of the normalised spec **and the
producer's `VERSION`** — e.g. `eval3ds-66aaff5b`, `act-old_eval-e2776116`. Bumping a
producer's `VERSION` changes every id it emits, so records made by different versions of
the eval code can never be silently mixed into one figure. Override spec fields on the
command line with `--set key=jsonvalue`:

```bash
python -m vnl_experiments.artifacts ensure --kind activations \
    --runs <ids> --set dataset='"new_eval"' --set limit_clips=8
```

The eval spec pins `checkpoint: "last"`, not a step number — which step is last cannot be
known before looking at the checkpoint directory, and a spec computable only where the
data lives would make "do I have this yet?" unanswerable from the laptop. The restored
step is recorded in `resolved.checkpoint_step`.

#### When to bump `VERSION`

The spec says *what was asked for*; `VERSION` says *how the producer makes it*. Together they
have to guarantee: same `(kind, wandb_id, spec_id)` ⇒ same recipe ⇒ bytes that may be pooled.

**Bump** when `produce()` could write different bytes for the same spec: a numeric bug fix, a
change in what is computed or recorded, or a change in the meaning of an existing field.

**Do not bump** when adding a spec key whose default is `None` (`normalise_spec` drops `None`
precisely so this stays valid — `EvalProducer.action_noise` is the precedent), when adding a
field to `resolved` (not hashed), or for refactors and performance work that cannot change
output.

A bump does not invalidate the committed analyses — it *partitions* history. `coverage_table`
looks up the `spec_id` pinned in `REQUIRES`, so a pinned folder keeps resolving its old files
at full coverage indefinitely; only a **bare**-kind requirement (`"eval"` with no `:id`)
re-resolves to the producer's current default and flips to 0/N. What the bump does change is
what `plan`/`ensure` consider current, which is the signal you want.

Two consequences worth planning for:

* Artifacts whose bytes the change provably cannot alter should be **adopted**, not
  recomputed: `artifacts adopt --kind K --runs R --from-spec <old-id>` hardlinks them onto the
  new id and records `producer.adopted_from`. It refuses any run where the predicate fails.
* A bump makes past damage *quiet*, since pinned folders keep rebuilding their old numbers and
  `--check` stays green. Locate the fallout deliberately with
  `artifacts audit-env --by-analysis` rather than waiting for a numeric diff.

The 2026-08-18 walker-XML fix (below) is the worked example: `eval`, `activations` and `video`
went to `VERSION = 2`, 363 + 16 + 16 artifacts were re-produced, 85 were adopted, and
`history` was untouched.

### Cluster workflow (the default for anything needing a checkpoint)

Checkpoints stay on the cluster; only results come down.

```bash
# laptop
python -m vnl_experiments.artifacts plan --kind eval --runs analysis/<q>/runs.csv --out todo.txt
scp todo.txt cluster:$SCRATCH/vnl-experiments/

# cluster
sbatch slurm_eval.sh todo.txt eval          # writes into $VNL_ARTIFACTS with sidecars

# laptop
export VNL_CLUSTER_ARTIFACTS=cluster:/n/holylfs06/.../artifacts
python -m vnl_experiments.artifacts pull --kind eval --runs analysis/<q>/runs.csv
```

`pull` rsyncs whole `<kind>/<wandb_id>/` directories and reindexes from the sidecars that
arrive with them. `plan` separates the runs it could produce locally from the ones that
need the cluster, so the split is explicit.

## 3. The question folder

```
analysis/<question-slug>/
├── extract.py          # the only script that reads the index or the store
├── runs.csv            # committed selection: which runs are in this analysis
├── data.csv            # committed snapshot — one row per run
├── curves.csv          # optional, from history artifacts
├── comparability.txt   # committed programmatic comparability report
├── coverage.txt        # committed per-condition artifact coverage
├── plot.py             # reads only the CSVs
├── figures/*.png  +  figures/manifest.json
└── report.md
```

Start from the template: `cp -r analysis/_template/<track> analysis/<track>/<question-slug>`.

### The freeze/refresh gate

`runs.csv` **is** the dataset definition; `CONDITIONS` in `extract.py` is the *query* that
produced it. Both are committed, and they are used in different modes:

| command | behaviour |
|---|---|
| `python extract.py` | **frozen** — rebuild `data.csv` from exactly the runs in `runs.csv`. Deterministic, no selector evaluation, no network. |
| `python extract.py --refresh` | re-run `CONDITIONS` against the index, print the added/removed/moved runs, rewrite `runs.csv` and `data.csv`. |
| `python extract.py --sync --refresh` | as above, refreshing the index from WandB first. |
| `python extract.py --check` | frozen rebuild diffed against the committed CSVs; non-zero exit on drift. |

So "redo this plot with exactly the same data" is the default and needs no thought, and
"now include the new runs" is one flag with a visible diff in the commit.

A condition selector is either a mapping of `index.select` filters or, when a cell needs
logic column equality cannot express, a callable taking the index frame and returning a
mask. A run matching two conditions raises: the cells are not mutually exclusive.

### Tiers and coverage

Each `extract.py` declares what it needs:

```python
REQUIRES = ["index", "history", "eval:eval3ds-66aaff5b"]
```

`pipeline.write_coverage` writes `coverage.txt` with per-condition counts, flags gaps
with `*** GAP ***`, names any *other* spec ids held for those runs (so "made by a
different eval version" doesn't read as "no data"), and prints the exact `plan`/`pull`
commands to close the gap.

**Runs are never silently dropped for a missing artifact.** A cohort quietly reported on
the subset that happened to have data is how `collision-model-xml` got written up with
0/32 offline evals.

### Two-stage rule

- **`extract.py` is the only code that reads the index or the artifact store**, and the
  only code that writes `data.csv`.
- **`plot.py` reads only the CSVs** in its folder. No `wandb`, no store, no network.

That separation is what lets a figure be restyled or re-rendered years later.

### Figure provenance

`plot.py` stamps each figure with a small footer — analysis name, input CSV hashes, repo
commit, date — and writes `figures/manifest.json` recording the same, so a figure that has
escaped into a slide deck can still be traced back to committed data. Set
`VNL_NO_FOOTER=1` for presentation figures; the manifest is written either way.

```python
from vnl_experiments.wandb_utils.style import apply_style, provenance, write_figure_manifest
```

Use `CONDITION_STYLE` / `color_for(condition)` / `marker_for(condition)` so a condition
keeps its colour and marker across every question, and `add_ms_axis(ax, max_delay)` for
delay plots.

### Videos in a report

Declare the runs to render, `ensure`/`pull` the `video` artifacts, and link them from
`report.md`. The mp4s stay in the (gitignored) store; commit a still or contact sheet into
`figures/` if the report needs to stand alone.

## 4. Comparability protocol (mandatory — programmatic *and* manual)

Before plotting two conditions together, confirm the runs are directly comparable.

**Programmatic** — `extract.py` calls `comparability_report(runs,
invariant_cols=INVARIANTS, group_col="condition")` and writes `comparability.txt`. It
lists the unique values of each invariant, overall and within each condition, flagging
anything that varies with `*** VARIES ***`. Include at minimum: `env`, `seed`, the network
sizes, the PPO settings, `summary._step` (**actual** trained steps, not just the
configured `total_steps`), and `git_commit`.

Since 2026-08-24 runs also log the **environment they ran in**, and a cohort that straddles
a cluster upgrade should treat these as invariants too:

| column | what it pins |
|---|---|
| `repos.nnx_ppo.commit`, `repos.vnl_playground.commit` | the algorithm and the task. `git_commit` covers only vnl-experiments; these two repos were previously unrecorded |
| `repos.*.dirty` | **any `True` here voids the commit** — the working copy had drifted, so the hash does not identify the code |
| `os`, `cuda_version`, `python` | kernel + glibc, CUDA toolkit, interpreter |

Artifact sidecars carry the matching `producer.repos`, `producer.cuda_driver`,
`producer.platform` and `producer.packages` (jax / mujoco / warp-lang / orbax / …),
because an artifact is produced long after its run and often on a different stack — a
run's WandB `requirements.txt` says nothing about the environment that later evaluated it.
A MuJoCo minor bump is not cosmetic: 3.6.0 vs 3.9.0 moves `old_eval` reward by ~3 % on an
identical checkpoint, against a delay-0-to-5 effect of only ~9 %.

**Manual** — the analyst additionally:

- inspects the actual run `config` independently (do **not** trust tags or notes alone);
- reads each run's `tags` and `notes`;
- when `git_commit` differs across runs, runs `git diff <a> <b>` and confirms the
  differences are additive and do not touch shared training, env, network or reward code;
- records the verdict in `report.md`.

Any unavoidable mismatch — a delay present in only one condition, a benign git difference,
a GPU-model split — is written up as a **caveat** in `report.md`.

## 5. `report.md`

See [`_template/report.md`](_template/rodent/report.md). It must contain the question, the
condition table, the coverage and comparability verdicts (including the manual half), the
figures with a sentence each on what to look at, hedged conclusions, and follow-ups.

## 6. Traps

Things that have already produced a wrong or nearly-wrong conclusion. Read before
starting. These are the ones that apply whatever is being trained; the traps peculiar to a
task live with it, in [`rodent/README.md`](rodent/README.md) and
[`dm_control_suite/README.md`](dm_control_suite/README.md), and are just as mandatory.

**`state == "finished"` silently drops runs that only died in the final eval.** The
2026-08-11 torque sweep — 46 runs, `ef060b73`, note *"New XML + reference_root."* — trained
all 600 M steps and then crashed in `run_final_eval`, so WandB records it as `failed`. It is
the only complete torque delay sweep on the new XML and reference frame (0 to 100 in 23
steps, `eff == delay`, one launch), and `position-control-open-loop/` was first written
without any of it, which is what left that analysis with no torque arm past delay 20 and a
headline it could not check. Gate on the property you mean instead:
`state == "finished" | summary._step >= config.ppo.total_steps`
(`pipeline`-free; see `completed_training` in `position-control-open-loop/extract.py`). That
is exactly discriminating on this XML + frame — of 126 non-finished runs there, those 46 are
the only ones reaching `total_steps`, and none has a `final_eval/*` key, which is the
signature. Such runs need an offline `eval` artifact to stand in for the missing inline eval,
so check coverage before selecting them; mixing an inline `final_eval` with a batch `eval`
in one figure costs under 0.5 % at the median (measured in
`position-control-open-loop/eval_calibration.txt`) but is still a mix, and belongs in a
`reward_source` column rather than in a reader's head.

**A `history` artifact made from a running run is a snapshot, and `ensure` will not replace
it.** The artifact key is `(kind, wandb_id, spec_id)` with no notion of how far the run had
got, so a `history` produced while a run was mid-training stays in the store, at that length,
for ever -- and `artifacts ensure` reports it as present. In
[`efference-copy-vs-proprioception/`](rodent/efference-copy-vs-proprioception/) four runs were
`running` at 110-190 M when their artifacts were first made and had reached 600 M by the next
session; rebuilding without noticing would have silently reported four *finished* runs as
having no usable readout, which looks exactly like more cluster attrition rather than like a
bug. The sidecar has what you need to detect it -- compare `resolved.max_step` against the
index's `summary._step` -- so check that before any rebuild that follows new runs finishing,
and re-produce the stale ones with `artifacts ensure --override`. The same applies to any
artifact kind whose content depends on how much of the run exists yet; an `eval` of a
`"last"` checkpoint has the same shape of problem, recorded in its own `resolved.checkpoint_step`.

**A common mid-training budget is biased against whatever trains slower.** Reading every
arm at a shared step count is the right way to include runs that died before the end -- but
it is only fair if the swept variable does not change the learning *rate*, and it often
does. In [`efference-copy-vs-proprioception/`](rodent/efference-copy-vs-proprioception/) the
efference queue is concatenated onto the decoder's input, so `efference_length` 100 means a
3 832-wide first layer instead of 108: at a 400 M readout the long-queue runs gain +14 % over
the next 200 M while the short-queue ones gain +0.1 %, so the early readout understates
exactly one end of the x-axis. The curve's *shape* there is an artefact; its *level* and the
between-condition ratio are not. Before quoting a shape from an early readout, plot the
(early -> final) gain against the swept variable for the runs that have both -- and check
whether the runs missing the final budget are correlated with the swept variable, since if
they are, that panel cannot be drawn where it is most needed.

**"Read the git diff and confirm it is benign" does not scale, so hash the load-bearing code
instead.** A cohort spanning a refactor can span 80 files and 10 000 lines with nothing
relevant changed. `efference-copy-vs-proprioception/code_identity.py` is the pattern: name
the handful of functions and files that actually build and run the networks, and hash them
at each commit in the cohort, comparing the source **with docstrings stripped and
re-unparsed** rather than raw bytes -- two of its four subjects differ in bytes only because
a script was renamed inside a docstring, and a raw-byte check would report a spurious
difference and train you to ignore it. Note what this cannot do: when `repos.*.dirty` is
`True` the commit does not identify the code that ran, so pair the hash with a group of runs
that are configured identically and differ only in commit/stack, and quote *that* spread.

**The cluster working copy drifts from the committed script.** The same rule — read
`env_params`, never the script at the run's commit — applies to *every* env knob. The
2026-07-09/07-19 new-XML cohort logged `body_target_frame = current_root` while the script
at their recorded commits (`201d6e11`, `0560d402`) said `reference_root`; the cluster tree
had been edited, a state only committed later as `456fbd7`. WandB stored no `diff.patch`,
so the logged config is the only record of what ran. Same for `torque_actuators` and
`walker_xml_path`. See [`collision-model-xml/`](rodent/collision-model-xml/). (`ef060b7`,
2026-08-11, has since set `reference_root` in both training scripts.)

**Eval reproducibility is host-dependent** (measured 2026-08-24). On the **cluster** the
offline eval is exactly reproducible: across 56 runs that hold two independently produced
artifacts of the same spec (the `action_noise` `None`-vs-`0` spec_id split minted duplicates
of `eval3ds-n00-*`), all 168 files are **byte-identical by sha256**, on both H200 and V100.
So `spec_id` -> bytes really is a function there, and a re-produced artifact that differs
from a stored one *is* evidence of a code or environment change.

On **this laptop** (RTX 4060) it is not: four identical `evaluate_run` invocations on one
checkpoint, full 169-clip `old_eval`, spanned 1.19 % of the mean in reward (1770.4 / 1791.6 /
1777.8 / 1790.5) and 5.5 steps in lifespan. The likely cause is memory-pressure-dependent XLA
autotuning under `XLA_PYTHON_CLIENT_PREALLOCATE=false` plus MuJoCo-warp atomics at low
occupancy; it was *not* reproduced on datacenter GPUs.

Consequences:

* **Produce artifacts on the cluster.** A laptop-produced eval is a ~1 % measurement of a
  quantity the cluster computes exactly, and mixing the two into one cohort injects noise
  that nothing in the manifest records. `producer.gpu` in the sidecar is what tells them
  apart.
* Do not use a laptop reproduction to decide whether a code change altered results --
  re-produce on the cluster, or compare the *network* instead (weights and forward outputs
  are deterministic everywhere), which is how the 2026-08-20 registry refactor was verified.
* `param_counts` is deterministic on every host; only the rollout metrics move.

**Never use `scan_history`.** The runs log ~7 300 iterations over ~50 keys; streaming that
for 80 runs ran over 40 minutes without finishing. Use the sampled endpoint
(`run.history(keys=[...], samples=N)`) — that is what the `history` artifact does, and it
returns the ~60-point eval series complete in well under a second per run. Ask only for
keys the run actually logged: with `keys=`, WandB drops rows where a requested key is
missing, so including a name the run never wrote can empty the frame. (For the same
reason the aligned multi-key history starts at the first step where *every* requested key
exists, which drops the step-0 point.)

**The logging API was renamed mid-project** (`episode_reward/mean` →
`eval/episode_reward/mean`, `lifespan_mean` → `eval/lifespan/mean`). A cohort spanning the
rename needs both fetched and coalesced — use `pipeline.first_present(row, new, old)`.
Note also that `eval_env = train_env` in these runs, so the logged reward is
eval-on-training-clips, not held-out.

**GPU model is a throughput confound.** Runs are scheduled across A100-SXM4-80GB and H200
nodes, ~1.6–1.8× apart, and node-to-node spread within one model is a few percent. Any
speed comparison must be restricted to one GPU model and matched on the experimental axis;
prefer medians over the history series to the final summary value, and drop the first
~10 % of samples (XLA compilation). Where the cluster cells disagree, add a controlled
local benchmark — see [`collision-model-xml/benchmark_xml.py`](rodent/collision-model-xml/benchmark_xml.py).

**The eval is not bit-reproducible.** MuJoCo Warp's GPU physics is nondeterministic, and
over a 502-step rollout that amplifies: re-evaluating the same checkpoint with the same
seed moves `episode_reward` by ~1 % and can flip individual clips between surviving and
terminating. Per-clip quantities are worst hit — every `std`, and `termination_rate` (at
32 clips one clip flipping is 3 pp). Don't read a small difference between two eval passes
as signal. `new_eval` is 32 clips, single seed: read its curves for trend, not point
values.

**The final training-curve point is not a measurement either.** The same nondeterminism
moves a single inline eval point (and hence the run summary's `eval/episode_reward/mean`)
by a few percent, more at short lifespans. Reduce a run to the **mean of the eval points
in the last 50 M steps** — five points, at the 10 M-step eval cadence — not to its last
one. This is not cosmetic: `collision-model-xml`'s headline PG-FM deficit at delay 50 is
−15.2 % from the final point and −7.9 % from the window, on the same two runs. The
independent noise bound is ±2.9 %, measured in
[`xml-ceiling-vs-convergence/`](rodent/xml-ceiling-vs-convergence/) from pairs of runs that share
a configuration.

**Inline and batch evals may measure different weights.** Training runs evaluate
themselves when they finish and push the headline numbers to the run summary under
`final_eval/…` (e.g. `final_eval/old_eval/episode_reward/mean`,
`final_eval/new_eval/termination_rate/survived`, `final_eval/params/total`), writing the
full record to `{ckpt_dir}/eval.json`. Runs before 2026-08-10 have no `final_eval/*` keys.
Since `0b4de48` training no longer saves an extra final checkpoint, so the inline eval
measures the **in-memory** network at `total_steps` while a later `eval_runs.py` pass
restores the newest checkpoint on disk. With `total_steps = 600M` and
`checkpoint_every_steps = 50M` these coincide; when they don't, the training script warns.
They are different artifact specs and must not be mixed in one figure. For any cross-run
comparison prefer a single batch re-evaluation of the whole cohort under one eval version
— and note that crashed, preempted and resumed runs never reach the inline eval, so
`eval_runs.py` remains the way to fill those gaps.

## 7. Plotting conventions

`plot.py` reads only the CSVs (§3) and calls `apply_style()` once. Beyond that, four
standing preferences. The first three are about honesty of the y-axis; the last is about a
condition meaning the same thing everywhere.

**Prefer raw reward on the y-axis.** Use a derived measure — fraction of maximum, a
difference, a percent of some baseline — only when it is unavoidable. Reward does not map
linearly onto competence: the distance from 50 to 100 is not the distance from 250 to 300,
nor the distance from 500 to 1000, so a ratio silently asserts a linearity the task does
not have. When a derived axis really is unavoidable, say in `report.md` in one line *why*
it was, and keep a raw-reward panel alongside it wherever the figure has room.

The usual reason to reach for a ratio is that the arms have different scales — different
tasks, different clip lengths, different reward weights. Prefer **small multiples**: one
panel per scale with raw reward on each y-axis and a shared x-axis. That answers "which is
better, and by how much" without asserting that a 10 % gain means the same thing in two
places.

**Name the reward source in the label, every time.** Which reward is being shown — the
training split, a held-out eval set, an out-of-distribution set, an inline end-of-training
number, an offline artifact — belongs in the axis label and in the report text, not in the
reader's memory. Use `style.reward_label(source)` so the phrasing is the same everywhere:

```python
ax.set_ylabel(reward_label("old_eval"))       # "Episode reward (held-out, old_eval)"
ax.set_ylabel(reward_label("dmc_eval"))       # "Episode reward (eval episodes, unscaled)"
```

Two of the worst bugs in §6 and in the per-track trap lists — the in-training `eval/*`
train-split bug, and the dm_control eval scaling — were invisible precisely because
"reward" was written where "which reward" was meant.

**Show every seed.** Where a condition has more than one seed (or any similar repetition),
draw the mean as a solid line and each seed as a thin, semi-transparent line, so the
reader sees the spread the mean was taken over rather than trusting it:

```python
from vnl_experiments.wandb_utils.style import plot_seeds
plot_seeds(ax, group, x="delay_k", y="reward_mean", seed_col="seed", condition="encdec")
```

`plot_seeds` averages replicates *within* a (seed, x) cell first, so each seed contributes
one curve and the mean weights seeds equally rather than weighting the seed that happened
to be run twice. It adds one "individual seed" proxy entry to the legend, not one per
seed. Collapsing seeds to a mean alone is only acceptable when the spread is reported
some other way, and a single-seed condition should not be drawn as if it were a mean.

**One condition, one colour, everywhere.** Add the condition to `CONDITION_STYLE` in
`wandb_utils/style.py` and use `color_for` / `marker_for` / `label_for`, rather than
defining a local colour dict — a local dict is how the same manipulation ends up two
colours in two figures. For a delay axis in physical units use
`add_ms_axis(ax, max_x, ctrl_dt_ms=...)`; the default is the rodent's 10 ms and the
control suite runs at 25.
