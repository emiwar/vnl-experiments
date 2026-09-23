"""Producers: the code that makes each kind of artifact.

A producer is a thin adapter around code that already exists elsewhere in the package --
it defines the artifact's *identity* (spec + version) and where the bytes land, and then
delegates. Adding a new artifact kind means adding a :class:`Producer` subclass and
listing it in :data:`PRODUCERS`; nothing else in the store needs to know about it.

``VERSION`` is part of every ``spec_id`` this producer emits. **Bump it whenever the
meaning of the output changes**, so that records made by the old and new code can never
end up in the same figure without anyone noticing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from vnl_experiments.artifacts.store import Entry, Store, producer_stamp, spec_id

REPO_ROOT = Path(__file__).resolve().parents[2]


class Producer:
    """Base class. Subclasses set the class attributes and implement :meth:`produce`."""

    KIND: str
    VERSION: int
    EXT: str
    #: Keys the spec may contain, with their defaults.
    DEFAULTS: dict[str, Any] = {}
    #: Set for producers that need the run's checkpoint, i.e. that can only run where
    #: the checkpoints live. ``plan`` uses this to explain what to send to the cluster.
    NEEDS_CHECKPOINT: bool = False
    #: Whether several runs may be produced concurrently. True only for producers that
    #: are pure I/O; anything touching the GPU stays serial.
    PARALLEL_SAFE: bool = False

    def spec(self, **overrides: Any) -> dict[str, Any]:
        """Fill a partial spec out with this producer's defaults."""
        unknown = set(overrides) - set(self.DEFAULTS)
        if unknown:
            raise KeyError(f"{self.KIND}: unknown spec keys {sorted(unknown)}; "
                           f"expected a subset of {sorted(self.DEFAULTS)}")
        return {**self.DEFAULTS, **overrides}

    def prefix(self, spec: Mapping[str, Any]) -> str:
        """Readable leading part of the ``spec_id``."""
        return self.KIND

    def spec_id(self, spec: Mapping[str, Any]) -> str:
        return spec_id(self.prefix(spec), spec, self.VERSION)

    def stamp(self) -> dict[str, Any]:
        return producer_stamp(type(self).__module__ + "." + type(self).__name__,
                              self.VERSION)

    def produce(self, wandb_id: str, spec: Mapping[str, Any], out_path: Path,
                ctx: Mapping[str, Any]) -> dict[str, Any]:
        """Write the artifact to ``out_path``; return the ``resolved`` facts block."""
        raise NotImplementedError

    def ensure(self, store: Store, wandb_id: str, spec: Mapping[str, Any], *,
               ctx: Mapping[str, Any] | None = None, override: bool = False) -> Entry:
        """Produce the artifact unless it is already in the store; record the sidecar."""
        sid = self.spec_id(spec)
        existing = store.lookup(self.KIND, wandb_id, sid)
        if existing is not None and not override:
            return existing
        out_path = store.path_for(self.KIND, wandb_id, sid, self.EXT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = self.produce(wandb_id, spec, out_path, ctx or {})
        return store.record(self.KIND, wandb_id, sid, out_path,
                            spec=spec, producer=self.stamp(), resolved=resolved)


# --------------------------------------------------------------------------------------
# history
# --------------------------------------------------------------------------------------


class HistoryProducer(Producer):
    """Sampled training history for one run, as a parquet frame.

    Uses WandB's **sampled** history endpoint (``run.history(keys=..., samples=...)``),
    never ``scan_history``. The runs log ~7 300 iterations over ~50 keys; streaming all
    of that took over 40 minutes for 80 runs without finishing, whereas the sampled
    endpoint returns the same curves in well under a second per run. ``samples`` is an
    upper bound on rows, and WandB returns fewer when a key is logged less often (the
    eval series is only ~60 points, and comes back complete).

    Passing an explicit ``keys`` list is what makes the returned rows *aligned* across
    metrics; with ``keys=None`` WandB samples each metric independently and the frame is
    mostly NaN.
    """

    KIND = "history"
    VERSION = 1
    PARALLEL_SAFE = True
    # csv.gz rather than parquet: the frames are tiny (tens to a couple of thousand
    # rows), pandas writes it with no extra dependency, and it stays greppable.
    EXT = ".csv.gz"
    DEFAULTS = {
        "keys": [
            # Reward/lifespan under both the old and the new logging names: the API
            # rename (`x` -> `eval/x`) happened mid-project, so a cohort spanning it
            # needs both fetched and coalesced at read time.
            "episode_reward/mean", "eval/episode_reward/mean",
            "lifespan_mean", "eval/lifespan/mean",
            "throughput/train_sps", "throughput/eval_sps",
        ],
        "samples": 2000,
        "project": "emiwar-team/nnx-ppo-rodent-delays",
    }

    def prefix(self, spec: Mapping[str, Any]) -> str:
        return f"hist{spec['samples']}"

    def produce(self, wandb_id: str, spec: Mapping[str, Any], out_path: Path,
                ctx: Mapping[str, Any]) -> dict[str, Any]:
        import wandb

        api = ctx.get("wandb_api") or wandb.Api(timeout=120)
        run = api.run(f"{spec['project']}/{wandb_id}")
        # Ask only for keys this run actually logged: with `keys=`, WandB returns rows
        # keyed on the *first* requested metric, so including a key the run never wrote
        # (the old-vs-new logging names never coexist) can empty the frame.
        logged = set(run.summary.keys())
        present = [k for k in spec["keys"] if k in logged]
        frame = run.history(keys=present, samples=spec["samples"], pandas=True) \
            if present else pd.DataFrame()
        frame.to_csv(out_path, index=False, compression="gzip")
        return {"rows": int(len(frame)),
                "columns": sorted(c for c in frame.columns if not c.startswith("_")),
                "max_step": int(frame["_step"].max()) if "_step" in frame and len(frame) else None}


# --------------------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------------------


class TimingProducer(Producer):
    """Per-iteration wall-clock series for one run: what the training loop spent time on.

    ``history`` answers "how did reward evolve"; this answers "where did the wall clock
    go". It records, for every PPO iteration the run logged, the three throughput gauges
    ``train_ppo`` emits and the wandb ``_runtime`` stamp of the row, from which
    :mod:`vnl_experiments.wandb_utils.timing` reconstructs a per-iteration budget
    (training / eval / video / checkpoint / everything else).

    Why this is a separate kind rather than a ``history`` spec override: ``history``
    issues **one** ``run.history(keys=[...])`` call, and WandB drops any row where a
    requested key is missing, so asking for the three gauges together returns only their
    intersection -- the ~80 rows on which a video happened to be rendered, i.e. 4 % of
    the iterations and none of the wall clock. The three series are logged at three
    different cadences and have to be fetched separately and joined on ``_step``, which
    is what this producer does.

    Two properties of the raw series matter enough to state here, because the artifact
    stores them uncorrected:

    * ``runtime_s`` is the row's wandb ``_runtime``, which is stamped when the row is
      *committed* -- not when it was logged. It therefore lags the iteration it belongs
      to by a fixed number of rows. The lag is recovered per run (and verified, to
      milliseconds) by :func:`~vnl_experiments.wandb_utils.timing.iteration_costs`; it is
      not baked in here, so a future change to how wandb commits rows does not
      retroactively corrupt stored artifacts.
    * Rows are one per *logged* iteration. The step-0 log (which carries the initial
      eval, video and checkpoint) has no ``train_sps``, so it is absent; everything
      before the first surviving row lands in the analysis's startup bucket.

    Cheap and network-only: three sampled-history calls, ~1.5 s per run, no checkpoint
    and no GPU. ``samples`` is an upper bound on rows and defaults high enough to return
    *every* iteration of a 2 G-step dm_control run (8 138 of them), because the
    reconstruction differences successive ``_runtime`` stamps and a sampled subset would
    silently merge the cost of the iterations it skipped.
    """

    KIND = "timing"
    VERSION = 1
    PARALLEL_SAFE = True
    EXT = ".csv.gz"
    #: The gauges, in the order the columns are written. Each is fetched on its own.
    GAUGES = {
        "train_sps": "throughput/train_sps",
        "eval_sps": "throughput/eval_sps",
        "video_sps": "throughput/video_sps",
    }
    DEFAULTS = {
        "samples": 20000,
        # A spec field, and deliberately the rodent project by default, exactly as for
        # `history`: the project is hashed into the spec_id, so a control-suite timing
        # artifact and a rodent one can never be pooled, and a dm_control analysis that
        # forgets the override fails loudly instead of reading the wrong project.
        "project": "emiwar-team/nnx-ppo-rodent-delays",
    }

    def prefix(self, spec: Mapping[str, Any]) -> str:
        return f"timing{spec['samples']}"

    def produce(self, wandb_id: str, spec: Mapping[str, Any], out_path: Path,
                ctx: Mapping[str, Any]) -> dict[str, Any]:
        import wandb

        api = ctx.get("wandb_api") or wandb.Api(timeout=120)
        run = api.run(f"{spec['project']}/{wandb_id}")
        logged = set(run.summary.keys())

        # `_runtime` rides along with the densest gauge (train_sps, one row per
        # iteration) rather than being fetched alone: asked for on its own it returns a
        # row for *every* log including step 0, which would not line up with the gauges.
        frame: pd.DataFrame | None = None
        present: list[str] = []
        for column, key in self.GAUGES.items():
            if key not in logged:
                continue
            want = [key, "_runtime"] if frame is None else [key]
            part = run.history(keys=want, samples=spec["samples"], pandas=True)
            if part.empty:
                continue
            part = part.set_index("_step").rename(columns={key: column,
                                                           "_runtime": "runtime_s"})
            present.append(column)
            frame = part if frame is None else frame.join(part, how="outer")

        if frame is None:
            frame = pd.DataFrame(columns=["_step", "runtime_s", *self.GAUGES])
        else:
            frame = frame.sort_index().reset_index()
            for column in self.GAUGES:
                if column not in frame:
                    frame[column] = float("nan")
            frame = frame[["_step", "runtime_s", *self.GAUGES]]
        frame.to_csv(out_path, index=False, compression="gzip")

        counts = {f"n_{c}": int(frame[c].notna().sum()) for c in self.GAUGES} \
            if len(frame) else {f"n_{c}": 0 for c in self.GAUGES}
        return {
            "rows": int(len(frame)),
            "gauges": present,
            "max_step": int(frame["_step"].max()) if len(frame) else None,
            "max_runtime_s": (float(frame["runtime_s"].max())
                              if len(frame) and frame["runtime_s"].notna().any()
                              else None),
            # A run whose rows were sampled rather than returned whole cannot be
            # differenced; `iteration_costs` refuses it, and this is how to see why.
            "truncated": bool(len(frame) >= spec["samples"]),
            **counts,
        }


# --------------------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------------------


def _asset_provenance(ckpt_dir: Path, env_class_hint: str | None) -> dict[str, str]:
    """Which XML files an offline rebuild of this checkpoint uses, for the sidecar.

    Re-derives the choice from the same inputs and the same function the env build uses
    (``config_io._choose``), so the stamp cannot drift from what was actually simulated.

    Recording this is not decoration. Until 2026-08-18 every offline rebuild silently
    replaced the run's ``walker_xml_path`` with the local default, so new-XML runs were
    re-simulated on ``rodent.xml``; nothing in the artifact said which body it used, which
    is why it went unnoticed for a month. An absent ``walker_xml_path`` in ``resolved``
    marks an artifact made before that fix.
    """
    from vnl_experiments.delays.evaluation import resolve_env_class
    from vnl_experiments.envs.config_io import local_xml_names

    config_path = Path(ckpt_dir) / "config.json"
    if not config_path.exists():
        return {}
    env_params = json.loads(config_path.read_text()).get("env_params", {})
    _, default_config_fn = resolve_env_class(env_class_hint or "", env_params,
                                             "AbsoluteImitation")
    return local_xml_names(env_params, default_config_fn())


class EvalProducer(Producer):
    """Offline re-evaluation of a run's checkpoint on the train / old_eval / new_eval sets.

    Delegates to :func:`vnl_experiments.delays.eval_runs.evaluate_run`, which is the same
    measurement code the training scripts run inline at the end of a run.

    The spec pins ``checkpoint: "last"`` rather than a step number, because which step is
    "last" cannot be known before looking at the checkpoint directory -- and a spec that
    can only be computed where the data lives would make "do I have this yet?"
    unanswerable from the laptop. The step that was actually restored is recorded in the
    sidecar's ``resolved.checkpoint_step``, and coverage checks assert on it.

    ``action_noise`` defaults to ``None`` rather than ``0.0`` on purpose: ``normalise_spec``
    drops ``None`` values, so adding this axis leaves the noise-free ``spec_id``
    (``eval3ds-66aaff5b``) and every artifact already made under it untouched.

    ``per_clip`` (2026-09-21) defaults to ``None`` for exactly the same reason, and is the
    second use of that pattern. Set, it keeps the ``[n_clips]`` vectors ``eval_dataset``
    already computes -- per-clip reward, lifespan, every termination flag, every reward
    term, every per-body error, plus the clip's behaviour label where the clip set has one
    -- under a ``per_clip`` key, and mints its own ``spec_id``. The published aggregates
    are bit-identical either way, which is what makes ``VERSION`` staying at 3 correct
    rather than convenient: ``normalise_spec`` drops the ``None``, so the default spec
    hashes exactly as before and the 393 artifacts already in the store keep resolving.
    ``evaluation_test.PerClipTest`` pins that equality.

    ``VERSION = 3`` (2026-08-24): ``_parse_net_params`` no longer truncates sub-1.0
    floats to zero, so the rebuilt network finally gets the ``latent_min_std`` its
    config specifies. The bottleneck samples at eval time, so a different latent
    ``std`` shifts the sampled latent and hence the actions (measured: mean
    |delta action| 5.4e-4, max 1.2e-2 on a real checkpoint; critic values unchanged).
    Small, but not zero, so the bytes are not poolable with ``VERSION = 2``.
    ``VERSION = 2`` (2026-08-18): the env rebuild no longer replaces the run's walker XML
    with the local default, so evals of runs trained on a non-default body now measure that
    body. This changes the numbers, hence a new ``spec_id``; ``eval3ds-66aaff5b`` and its
    noise variants remain readable for the analyses that pinned them.
    """

    KIND = "eval"
    VERSION = 3
    EXT = ".json"
    NEEDS_CHECKPOINT = True
    DEFAULTS = {
        "datasets": ["train", "old_eval", "new_eval"],
        "checkpoint": "last",
        "seed": 0,
        "limit_clips": None,
        "new_eval_h5": "eval_clips_32x30s.h5",
        # Std of a fixed Gaussian perturbation added to the executed action
        # (post-tanh, clipped to [-1, 1]). None = the ordinary noise-free eval.
        "action_noise": None,
        # Keep the per-clip vectors alongside the aggregates. None = aggregates only,
        # which is the historical record shape. See the class docstring.
        "per_clip": None,
    }

    def spec(self, **overrides: Any) -> dict[str, Any]:
        spec = super().spec(**overrides)
        # `--set action_noise=0` parses as an int, and json.dumps writes `0` where `0.0`
        # writes `0.0`, so the two hash to different spec_ids for a numerically identical
        # eval. That silently minted a parallel `eval3ds-n00-21b2d9a8` alongside the real
        # n00 spec on 2026-08-19. produce() uses the value numerically, so canonicalise it
        # here. Float values are unaffected, so no existing spec_id changes.
        if spec.get("action_noise") is not None:
            spec["action_noise"] = float(spec["action_noise"])
        # Same hazard, one axis over: `--set per_clip=false` produces bytes identical to
        # the default, but `False` survives `normalise_spec` where `None` is dropped, so
        # it would mint a duplicate of `eval3ds-382e9e69` under a second id. Unlike
        # `action_noise`, where an explicit 0.0 is a real sweep point distinct from "no
        # noise", there is nothing for a falsy `per_clip` to mean other than the default --
        # so collapse it rather than canonicalising it to a separate value.
        if not spec.get("per_clip"):
            spec["per_clip"] = None
        return spec

    def prefix(self, spec: Mapping[str, Any]) -> str:
        base = f"eval{len(spec['datasets'])}ds"
        noise = spec.get("action_noise")
        # `is None`, not truthiness: an explicit 0.0 is a sweep point and must read
        # as `n00` rather than borrow the noise-free prefix.
        if noise is not None:
            base = f"{base}-n{round(noise * 100):02d}"
        # Cosmetic only -- the digest is over the whole spec either way -- but it makes a
        # per-clip artifact identifiable in a directory listing, which matters because
        # these are ~100x the size of an aggregate-only record.
        return f"{base}-pc" if spec.get("per_clip") else base

    def produce(self, wandb_id: str, spec: Mapping[str, Any], out_path: Path,
                ctx: Mapping[str, Any]) -> dict[str, Any]:
        from vnl_experiments.delays.eval_runs import evaluate_run
        from vnl_experiments.delays.evaluation import DEFAULT_NEW_EVAL_H5

        ckpt_dir = ctx.get("checkpoint_dir")
        if ckpt_dir is None:
            raise FileNotFoundError(
                f"no checkpoint found for {wandb_id}; evals must run where the "
                f"checkpoints live (see `artifacts plan --kind eval`)"
            )
        new_eval_h5 = DEFAULT_NEW_EVAL_H5.with_name(spec["new_eval_h5"])
        record = evaluate_run(wandb_id, ctx.get("wandb_name", wandb_id),
                              ctx.get("env_class", "AbsoluteImitation"),
                              Path(ckpt_dir), new_eval_h5,
                              spec["seed"], spec["limit_clips"],
                              action_noise=spec.get("action_noise"),
                              datasets=tuple(spec["datasets"]),
                              per_clip=bool(spec.get("per_clip")))
        if record is None:
            raise RuntimeError(f"evaluate_run returned nothing for {wandb_id}")
        out_path.write_text(json.dumps(record, indent=2))
        # `evaluate_networks` returns a flat record: step / env_class / network_class at
        # the top level, the per-dataset measurements under "datasets".
        return {"checkpoint_step": record.get("step"),
                "env_class": record.get("env_class"),
                "network_class": record.get("network_class"),
                "datasets": sorted(record.get("datasets", {})),
                # `resolved` is not hashed, so this is free, and `manifest_df`
                # surfaces it as a filterable `resolved.action_noise` column.
                "action_noise": record.get("action_noise"),
                "per_clip": spec.get("per_clip"),
                **_asset_provenance(Path(ckpt_dir), ctx.get("env_class"))}


class ActivationsProducer(Producer):
    """Per-layer unit activations for one run on one dataset, as HDF5.

    Delegates to :func:`vnl_experiments.delays.record_activations.record_run`. One
    artifact per *dataset* (not one per run), because these files are 1--2 GB each and
    an analysis usually wants only one dataset -- keeping them separate is what makes
    selective pulling from the cluster worth doing.

    These are also the artifacts most worth reusing across questions: recording them
    costs a full rollout with recording hooks, and the result is question-independent.

    ``VERSION = 3`` (2026-08-24): ``_parse_net_params`` no longer truncates sub-1.0
    floats to zero, so the rebuilt network finally gets the ``latent_min_std`` its
    config specifies. The bottleneck samples at eval time, so a different latent
    ``std`` shifts the sampled latent and hence the actions (measured: mean
    |delta action| 5.4e-4, max 1.2e-2 on a real checkpoint; critic values unchanged).
    Small, but not zero, so the bytes are not poolable with ``VERSION = 2``.
    ``VERSION = 2`` (2026-08-18): the env rebuild no longer replaces the run's walker XML
    with the local default. Recordings of runs trained on a non-default body were made on
    the wrong body, off the policy's state distribution, so they are not comparable with
    the ones made after the fix -- hence a new ``spec_id`` rather than an in-place repair.
    """

    KIND = "activations"
    VERSION = 3
    EXT = ".h5"
    NEEDS_CHECKPOINT = True
    DEFAULTS = {
        "dataset": "old_eval",
        "checkpoint": "last",
        "seed": 0,
        "limit_clips": None,
        "max_steps": None,
        "new_eval_h5": "eval_clips_32x30s.h5",
    }

    def prefix(self, spec: Mapping[str, Any]) -> str:
        return f"act-{spec['dataset']}"

    def produce(self, wandb_id: str, spec: Mapping[str, Any], out_path: Path,
                ctx: Mapping[str, Any]) -> dict[str, Any]:
        import h5py

        from vnl_experiments.delays.evaluation import DEFAULT_NEW_EVAL_H5
        from vnl_experiments.delays.record_activations import record_run

        ckpt_dir = ctx.get("checkpoint_dir")
        if ckpt_dir is None:
            raise FileNotFoundError(
                f"no checkpoint found for {wandb_id}; activations must be recorded "
                f"where the checkpoints live")

        name = ctx.get("wandb_name", wandb_id)
        # record_run names its output <name>__<dataset>.h5 in output_dir; give it a
        # private directory and move the result onto the store path afterwards.
        staging = out_path.parent / f".staging-{out_path.name}"
        staging.mkdir(parents=True, exist_ok=True)
        try:
            record_run(name, ctx.get("condition", ""),
                       ctx.get("env_class", "AbsoluteImitation"), Path(ckpt_dir),
                       [spec["dataset"]],
                       DEFAULT_NEW_EVAL_H5.with_name(spec["new_eval_h5"]),
                       spec["seed"], spec["limit_clips"], ctx.get("clip_chunk", 16),
                       spec["max_steps"], staging, True)
            produced = staging / f"{name}__{spec['dataset']}.h5"
            if not produced.exists():
                raise RuntimeError(f"record_run wrote nothing for {wandb_id}")
            produced.replace(out_path)
        finally:
            for leftover in staging.glob("*"):
                leftover.unlink()
            staging.rmdir()

        with h5py.File(out_path, "r") as handle:
            attrs = dict(handle.attrs)
            # Layer names contain "/", so h5py stores them as nested groups; count the
            # leaf datasets rather than the top-level keys.
            layers: list[str] = []
            handle["activations"].visititems(
                lambda name, obj: layers.append(name)
                if isinstance(obj, h5py.Dataset) else None)
        return {"checkpoint_step": int(attrs.get("step", 0)) or None,
                "dataset": spec["dataset"],
                "n_clips": int(attrs.get("n_clips", 0)) or None,
                "n_steps": int(attrs.get("n_steps", 0)) or None,
                "n_layers": len(layers),
                **_asset_provenance(Path(ckpt_dir), ctx.get("env_class"))}


class VideoProducer(Producer):
    """A rendered rollout video for one run, to put next to the numbers in a report.

    Delegates to :func:`vnl_experiments.delays.eval_videos.render_run`. Alongside the
    mp4 it leaves ``.h5`` (rollout + reference qpos at frame rate, enough to re-render
    or overlay without re-simulating) and ``.stats.json`` in the same directory.

    ``auto_reset=False`` -- the default -- keeps simulating past a termination so the
    failure mode is visible; ``True`` snaps back to the reference so the timeline stays
    locked to the mocap clip.

    ``VERSION = 3`` (2026-08-24): ``_parse_net_params`` no longer truncates sub-1.0
    floats to zero, so the rebuilt network finally gets the ``latent_min_std`` its
    config specifies. The bottleneck samples at eval time, so a different latent
    ``std`` shifts the sampled latent and hence the actions (measured: mean
    |delta action| 5.4e-4, max 1.2e-2 on a real checkpoint; critic values unchanged).
    Small, but not zero, so the bytes are not poolable with ``VERSION = 2``.
    ``VERSION = 2`` (2026-08-18): the env rebuild no longer replaces the run's walker XML
    with the local default, so videos of runs trained on a non-default body now show that
    body. Earlier renders of those runs show the wrong animal.
    """

    KIND = "video"
    VERSION = 3
    EXT = ".mp4"
    NEEDS_CHECKPOINT = True
    DEFAULTS = {
        "n_clips": 4,
        "fps": 50,
        "auto_reset": False,
        "checkpoint": "last",
        "seed": 0,
        "new_eval_h5": "eval_clips_32x30s.h5",
    }

    def prefix(self, spec: Mapping[str, Any]) -> str:
        return f"vid{spec['n_clips']}c{'-reset' if spec['auto_reset'] else ''}"

    def produce(self, wandb_id: str, spec: Mapping[str, Any], out_path: Path,
                ctx: Mapping[str, Any]) -> dict[str, Any]:
        from vnl_experiments.delays.eval_videos import NEW_EVAL_H5, render_run

        ckpt_dir = ctx.get("checkpoint_dir")
        if ckpt_dir is None:
            raise FileNotFoundError(
                f"no checkpoint found for {wandb_id}; videos must be rendered where "
                f"the checkpoints live")

        # render_run writes rollout<suffix>.{mp4,h5} + stats<suffix>.json into a
        # directory; point it at the store directory and use the spec_id as the stem so
        # its three outputs sit together under one identity.
        stem = out_path.name[: -len(self.EXT)]
        stats = render_run(
            Path(ckpt_dir), out_path.parent,
            n_clips=spec["n_clips"], fps=spec["fps"], auto_reset=spec["auto_reset"],
            seed=spec["seed"],
            new_eval_h5=str(Path(NEW_EVAL_H5).with_name(spec["new_eval_h5"])),
            suffix="",
        )
        for made, wanted in [("rollout.mp4", out_path),
                             ("rollout.h5", out_path.with_suffix(".h5")),
                             ("stats.json", out_path.with_suffix(".stats.json"))]:
            source = out_path.parent / made
            if source.exists():
                source.replace(wanted)
        return {"checkpoint_step": stats.get("step"),
                "env_class": stats.get("env_class"),
                "network_class": stats.get("network_class"),
                "n_clips": stats.get("n_eval_clips"),
                "stem": stem,
                **_asset_provenance(Path(ckpt_dir), stats.get("env_class"))}


class TraceProducer(Producer):
    """Per-step traces of one run's rollout on one dataset, as HDF5.

    Delegates to :func:`vnl_experiments.delays.eval_runs.trace_run`, which rebuilds the
    checkpoint through the same path ``eval`` does.

    **Why this is not part of ``eval``.** The eval record answers "how well did it do on
    this clip". A trace answers "what was it doing at this moment", which is the only way
    to resolve reward or failure by *behaviour* inside a clip that spans several -- the
    30 s ``new_eval`` clips average 13 MotionMapper behaviours each, so any per-clip
    number there is a mixture. That needs arrays, so HDF5 rather than JSON, and it is
    wanted for one dataset at a time, so one artifact per (run, dataset) -- which is
    ``activations``'s shape, not ``eval``'s.

    Contents: every env metric plus ``reward``, ``alive`` and ``running``, each
    ``[n_clips, n_steps]`` float32, gzipped. Roughly 3--10 MB per run per dataset.
    Post-termination steps are zeroed and flagged; ``alive`` is the mask to apply to the
    metrics and ``running`` is the one that sums to ``lifespan_steps`` -- see
    :func:`vnl_experiments.delays.evaluation._trace_rollout` for why those differ by one
    step. ``current_frame`` is traced, so the reference frame a step was tracking is read
    rather than inferred from the step index.

    ``VERSION = 1`` (2026-09-21): new kind.
    """

    KIND = "trace"
    VERSION = 1
    EXT = ".h5"
    NEEDS_CHECKPOINT = True
    DEFAULTS = {
        # `new_eval` by default: it is the only dataset with per-frame behaviour labels
        # available, and its 30 s clips are the ones a per-clip number cannot describe.
        "dataset": "new_eval",
        "checkpoint": "last",
        "seed": 0,
        "limit_clips": None,
        "max_steps": None,
        "new_eval_h5": "eval_clips_32x30s.h5",
    }

    def prefix(self, spec: Mapping[str, Any]) -> str:
        return f"trace-{spec['dataset']}"

    def produce(self, wandb_id: str, spec: Mapping[str, Any], out_path: Path,
                ctx: Mapping[str, Any]) -> dict[str, Any]:
        from vnl_experiments.delays.eval_runs import trace_run
        from vnl_experiments.delays.evaluation import DEFAULT_NEW_EVAL_H5

        ckpt_dir = ctx.get("checkpoint_dir")
        if ckpt_dir is None:
            raise FileNotFoundError(
                f"no checkpoint found for {wandb_id}; traces must be recorded where the "
                f"checkpoints live (see `artifacts plan --kind trace`)")

        meta = trace_run(wandb_id, ctx.get("wandb_name", wandb_id),
                         ctx.get("env_class", "AbsoluteImitation"),
                         Path(ckpt_dir),
                         DEFAULT_NEW_EVAL_H5.with_name(spec["new_eval_h5"]),
                         spec["seed"], spec["limit_clips"], spec["max_steps"],
                         spec["dataset"], out_path)
        if meta is None:
            raise RuntimeError(f"trace_run returned nothing for {wandb_id}")
        return {**meta, **_asset_provenance(Path(ckpt_dir), ctx.get("env_class"))}


PRODUCERS: dict[str, Producer] = {
    p.KIND: p() for p in (HistoryProducer, TimingProducer, EvalProducer,
                          ActivationsProducer, VideoProducer, TraceProducer)
}


def get_producer(kind: str) -> Producer:
    if kind not in PRODUCERS:
        raise KeyError(
            f"no producer registered for {kind!r}. Registered: {sorted(PRODUCERS)}. "
            f"Artifacts of other kinds can still be pulled, verified and indexed."
        )
    return PRODUCERS[kind]
