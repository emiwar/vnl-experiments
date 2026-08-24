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
                     analysis/<question>/          (§3 the question folder)
                       extract.py → data.csv → plot.py → figures/ → report.md
```

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

### The four kinds

| kind | what | cost | where it can run |
|---|---|---|---|
| `history` | sampled training curves + throughput, per run | seconds | anywhere (WandB) |
| `eval` | offline re-evaluation on train / `old_eval` / `new_eval` | minutes, GPU | needs a checkpoint |
| `activations` | per-layer unit activations on one dataset | minutes, 1–2 GB each | needs a checkpoint |
| `video` | rendered rollout mp4 + qpos h5 + stats | minutes, GPU | needs a checkpoint |

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

Start from the template: `cp -r analysis/_template analysis/<question-slug>`.

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

See [`_template/report.md`](_template/report.md). It must contain the question, the
condition table, the coverage and comparability verdicts (including the manual half), the
figures with a sentence each on what to look at, hedged conclusions, and follow-ups.

## 6. Traps

Things that have already produced a wrong or nearly-wrong conclusion. Read before
starting.

**`body_target_frame` lives on the env, not the network.** `AbsoluteImitation` reads
`config["env_params"]["body_target_frame"]`. The copy under
`config["net_params"]["body_target_frame"]` is **inert** — the training scripts set it on
`net_config`, where nothing reads it, and it was only logged. Every AbsoluteImitation run
before 2026-07-06 therefore trained with `current_root` regardless of what `net_params`
shows. Always filter and label on `env_params`. This invalidated the reference-root vs
current-root comparison in [`imitation-target-representation/`](imitation-target-representation/).

**A decoder-input ablation looks exactly like the standard efference baseline.**
`net_params.dec_use_intention=False` / `dec_use_proprioception=False` (added 2026-08-21)
drop the intention or the proprioception stream from the enc-dec decoder. Such a run keeps
the standard `{enc,dec,critic}_hidden_sizes`, keeps `efference_length == delay_k`, and
carries the `TrainEvalSplit` tag — so it passes every test the "standard efference
baseline" cohorts apply and joins them silently. Every folder that selects that cohort now
gates on `pipeline.full_decoder_inputs(net)` (dict form, for the live-fetch extractors) or
`& pipeline.full_decoder_inputs_mask(df)` (index form). Use those rather than a plain
filter kwarg: the columns are **absent**, not `True`, on every run predating the flags, so
`index.select(..., **{"net_params.dec_use_intention": True})` would drop the entire history.
The run name and tags also carry a `nointent` / `noproprio` token.

**The cluster working copy drifts from the committed script.** The same rule — read
`env_params`, never the script at the run's commit — applies to *every* env knob. The
2026-07-09/07-19 new-XML cohort logged `body_target_frame = current_root` while the script
at their recorded commits (`201d6e11`, `0560d402`) said `reference_root`; the cluster tree
had been edited, a state only committed later as `456fbd7`. WandB stored no `diff.patch`,
so the logged config is the only record of what ran. Same for `torque_actuators` and
`walker_xml_path`. See [`collision-model-xml/`](collision-model-xml/). (`ef060b7`,
2026-08-11, has since set `reference_root` in both training scripts.)

**Every offline rebuild used to silently swap the walker XML** (fixed 2026-08-18). A
checkpoint records the *cluster* path of its XML, which does not exist on a laptop, so
`parse_env_config` repaired it — by taking the local default, `consts.RODENT_XML_PATH` =
`rodent.xml`. A run trained on `rodent_no_tail_collisions.xml` was therefore re-simulated on
a **different body**, in three independent copies of the same block
(`delays/evaluation.py`, `delays/eval_videos.py`, `tools/checkpoint_utils.py`) and so in
every `eval`, `activations` and `video` artifact of the new-XML cohort. The inline
end-of-training eval was unaffected — it is handed the live `env_config` — which is exactly
why the discrepancy hid: offline `old_eval` reward sat 2 % / 13 % / 27 % / 42 % below inline
at delays 0 / 10 / 20 / 50, with survival 0.23 vs 0.67 at delay 50, and nothing in the
artifact said which body it had used.

Three lasting consequences:

* `envs/config_io.resolve_local_xml_paths` now repairs the *directory* while keeping the
  *file*, and warns loudly when the run's own XML is genuinely unavailable locally.
* Producers stamp `resolved.walker_xml_path` / `arena_xml_path`. **An absent stamp means
  pre-fix**, which is what makes the damage decidable after the fact.
* `artifacts audit-env` classifies every stored artifact as broken / adoptable / repaired
  and writes the re-production run lists. It found 363 eval, 16 activation and 16 video
  artifacts built on the wrong body; the analyses resting on them were
  [`action-noise-robustness/`](action-noise-robustness/) (all six eval specs),
  [`collision-model-xml/`](collision-model-xml/) (67/149 runs) and
  [`xml-ceiling-vs-convergence/`](xml-ceiling-vs-convergence/) (16/29). It reports zero once a
  folder is repointed, so it doubles as the done-check — it reads *pinned* spec ids only, not
  every id a script mentions, so a retirement note in a comment does not keep a fixed folder
  flagged.

The general lesson is narrower than "check your paths": **a reconstruction that repairs an
input must record what it chose.** Any field a rebuild silently substitutes is a field no
downstream analysis can audit.

A second lesson, from how long this survived: **verifying provenance against the record that
was already trusted is not verification.** `collision-model-xml/report.md` stated that "every
eval was verified to have used the run's own body and frame" — and it had been, by comparing
the run's stored config against itself. Nothing in that check could see what the eval process
actually loaded. Post-fix, the equivalent check reads `resolved.walker_xml_path` off the
artifact and compares it to `env_params.walker_xml_path` from the index — two independently
written records. `assert_artifact_body` in `collision-model-xml/extract.py` and
`xml-ceiling-vs-convergence/extract.py` is the pattern; copy it into any folder that spans two
bodies.

**Outcome, 2026-08-19.** [`collision-model-xml/`](collision-model-xml/) carries a retraction:
its headline result ("the new body falls over far more often; survival collapses") was the
artefact, and the corrected data reverses it. The v1 evals understated new-XML held-out reward
by 3.8 % at delay 0 rising to 77 % at delays 90–100, while the 79 old-XML runs are
bit-identical across the fix — so the bug acted on exactly one arm of every contrast in that
folder. [`xml-ceiling-vs-convergence/`](xml-ceiling-vs-convergence/) went the other way: its
curve-based primary result never moved (curves come from `history`), and the newly-usable evals
gave it a held-out measurement it had been missing on 25 of 29 runs.

**The in-training `eval/*` series measured the *train* split** (fixed 2026-08-20). Both
delays training scripts built a held-out env and then threw it away:

```python
eval_env = AbsoluteImitation(env_config, clips=test_clips)
eval_env = train_env          # <- train_rodent_delays.py:173, train_rodent_forward_model.py:188
```

The override dates from the file's creation (`e5bbf3f`, 2026-06-01), so **every delays run
to date** reported in-training eval on the clips it was training on. What that touches:

* **Affected: the WandB `eval/*` metrics** — `eval/episode_reward/*`, `eval/lifespan/*`,
  `eval/net/*`, `eval/env/*`. These are what the `history` artifact captures, so any
  curve-based result is a train-split learning curve, not a generalisation measurement.
* **Not affected: the offline `eval` artifacts, nor the inline end-of-training eval**
  (`evaluation.run_final_eval`). Both are handed `train_clips` / `test_clips` explicitly and
  build their own envs via `build_datasets`, so their `train` / `old_eval` / `new_eval`
  datasets were always genuinely separated. Held-out numbers taken from the store are sound.

The comparability consequence belongs with §4: `eval/*` means *train-split* before the fix
and *held-out* after it. **Never pool or contrast `eval/*` across 2026-08-20** — a post-fix
run will look worse for reasons that have nothing to do with what it is testing. Add
`created_at` (or `git_commit`) to `INVARIANTS` in any folder whose runs straddle the date,
and prefer the offline `eval` artifacts for anything held-out.

Action: audit any question folder whose primary result rests on `history`-derived `eval/*`
curves. [`xml-ceiling-vs-convergence/`](xml-ceiling-vs-convergence/) is the first to check —
§6 already records that its primary result is curve-based from `history` — and the check is
whether "held-out" is claimed anywhere for a number that came from a curve. The fix itself
changes no stored artifact and needs no `VERSION` bump.

**19 runs trained with the regularisation switched off** (2026-08-21 to 2026-08-24).
`delays.network_builders._parse_net_params` ran `int(v)` on every value, and
`int(0.01) == 0` in Python, so every sub-1.0 float was truncated to zero. The parser
existed to decode the stringified `config.json` on the **eval** side, where those four
values only touch a regularisation term and a reported metric. The 2026-08-21 registry
refactor put **training** on the same path, and from then until the fix every run trained
with:

| net_param | config says | actually used |
|---|---|---|
| `entropy_weight` | 0.01 | **0** — no entropy bonus |
| `kl_weight` | 0.001 | **0** — no KL penalty |
| `min_std` | 0.1 | **0** — no policy-std floor |
| `latent_min_std` | 0.01 | **0** |

The result is premature entropy collapse: on `old_eval` the action std fell 0.167 -> 0.049
(below the floor that was supposed to be enforced — that impossibility is what exposed the
bug), the bottleneck KL rose 21.7 -> 146.0, `root_too_far` terminations went 1.2 % -> 18.3 %,
and reward fell **14 % at delay 0 and 36 % at delay 10** against the matched 2026-08-11
runs. It hits every architecture equally, so the affected LSTM runs say nothing about
recurrence.

**The logged config does not show it.** `net_params` records the intended 0.01 / 0.001 /
0.1 — the truncation happens after the config is written — so no column distinguishes a
broken run and no comparability report can flag one. The discriminator is the commit:
`pipeline.UNREGULARIZED_COMMITS`, with `pipeline.regularized_training_mask(df)` as the row
mask. `created_at` is **not** a safe proxy: `dgmexgcj` trained on 2026-08-21 from the
earlier `afbeea0` and is fine.

The general lesson is the sibling of the walker-XML one: **a value that is recorded before
it is transformed is not a record of what ran.** `env_params` is authoritative because the
env reads it directly; `net_params` was not, because a parser sat between it and the
network. Where a config passes through coercion, the thing worth asserting is the value the
*module* ended up with — which is what `network_builders_test.ParseNetParamsTest` now does.

`eval`, `activations` and `video` went to `VERSION = 3` on 2026-08-24, because the fix
restores `latent_min_std` and the bottleneck samples at eval time, shifting the actions
slightly (mean |delta| 5.4e-4, max 1.2e-2; critic values unchanged). Old artifacts stay
valid at their pinned `spec_id`s and need re-producing only for a formal old-vs-new
comparison. `history` is unaffected and keeps `VERSION = 1`.

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
local benchmark — see [`collision-model-xml/benchmark_xml.py`](collision-model-xml/benchmark_xml.py).

**Two clocks in the eval datasets.** `clip_length` is in **mocap frames @ 50 Hz** (250 →
5 s, 1500 → 30 s), but the policy runs at `ctrl_dt = 0.01 s` (100 Hz). Rollouts are
`ceil(frames / (ctrl_dt·mocap_hz)) + 2` control steps — 502 for train/`old_eval`, 3002 for
`new_eval`. `lifespan_steps` and delays are in control steps (1 step = 10 ms).

The three eval datasets are: **`train`**, the 80 % training split (the clips the policy
trained on); **`old_eval`**, the held-out 20 % split (unseen clips, same 250-frame length);
and **`new_eval`**, 32 fresh 1500-frame clips. Each record carries, per dataset,
`episode_reward`, `lifespan_steps`, per-reason `termination_rate` (incl. `survived`),
per-step `errors` and network `net_metrics` (e.g. `fm_pred_mse`), plus hierarchical
`param_counts`.

**Raw reward is comparable within a dataset, not across.** Cumulative `episode_reward`
scales with clip length (~6× on `new_eval`). Across datasets use `reward_per_step` and the
per-second `hazard_rate` = `(1 − survived) / mean-alive-time` (failure terminations only;
end-of-clip truncations are censored, not events). Prefer the hazard to a raw survival
fraction: survival penalises longer clips for having more chances to fail.

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
[`xml-ceiling-vs-convergence/`](xml-ceiling-vs-convergence/) from pairs of runs that share
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

## 7. The legacy directories

`eval_results/eval_results/` (263 batch eval JSONs), `eval_results/old_eval_results/` (an
older batch) and `eval_results/activations/` (13 files, 22 GB, keyed by run *name*) have
been adopted into the store by `artifacts import-legacy`, **hardlinked** so nothing was
duplicated. They appear under fixed, unhashed spec ids — `legacy-batch`, `legacy-batch-v0`,
`legacy-<dataset>` — because their true specs were never recorded and inventing a hash
would imply a precision that does not exist. The originals can be deleted once
`artifacts verify` is clean; `eval_runs.py --collect` still gathers inline `eval.json`
files into `eval_results/eval_results/`, so re-run `import-legacy` after a collect.

Analyses written before this pipeline (everything except
[`collision-model-xml/`](collision-model-xml/),
[`xml-ceiling-vs-convergence/`](xml-ceiling-vs-convergence/),
[`explicit-vs-implicit-fm-2g/`](explicit-vs-implicit-fm-2g/) and
[`explicit-vs-implicit-fm-budgets/`](explicit-vs-implicit-fm-budgets/)) still read
`eval_results/`
directly and fetch from WandB in their `extract.py`. They remain valid and their CSVs are
unchanged; convert one to the layout above when you next need to touch it.
