"""Golden test: rebuilding a config from a checkpoint's ``config.json`` must not drift.

``config.json`` is the on-disk contract that lets a trained checkpoint be reloaded and
evaluated long after the run. Several pieces of code reconstruct a live config from it --
:func:`evaluation.parse_env_config` for the env, :func:`network_builders._parse_net_params`
for the network -- and they are the only thing standing between an old checkpoint and
being unloadable. They are also easy to break silently: a wrong value here does not raise,
it evaluates a *different model* or simulates a *different body* and reports a number.

Both failure modes have happened. A bare ``int(v)`` in ``_parse_net_params`` truncated
every sub-1.0 float to zero (``entropy_weight`` 0.01, ``kl_weight`` 0.001, ``min_std``
0.1), and a naive asset-path repair re-simulated runs on ``rodent.xml`` instead of the
``rodent_no_tail_collisions.xml`` they trained on -- worth up to 42 % of the reward.

So this test pins the *current* output byte-for-byte. It is not a correctness proof; it is
a change detector. If a refactor changes any value here, the diff says exactly which field
moved, and the question "is that intended?" gets asked before the numbers are believed.

``testdata/configs/*.json`` are real ``config.json`` files copied out of
``checkpoints/`` and ``downloaded_checkpoints/``, so the test is hermetic and runs on any
machine. They cover the variants that actually differ:

* ``encdec_modern`` -- current ``AbsoluteImitation`` run, cluster asset paths;
* ``encdec_no_btf`` -- predates ``body_target_frame``, so it exercises the
  env-class heuristic's ``Imitation`` branch;
* ``forward_model_recent`` -- the forward-model architecture;
* ``legacy_stringified`` -- everything written as strings (``"512"``, ``"True"``,
  ``"None"``) by the old ``json.dump(..., default=str)``, plus the extra top-level keys
  the distillation scripts wrote;
* ``nervenet_sequential`` -- ``network_class`` recorded as ``"<class '...'>"`` rather than
  a bare name, which is what the substring branch of ``get_architecture`` exists for;
* ``recurrent_synthetic`` -- built from ``recurrent_defaults()`` because no recurrent
  checkpoint is held locally; pins the recurrent net-params keys all the same.

Regenerate the expectations deliberately, never casually::

    python -m vnl_experiments.delays.reload_golden_test --regenerate

and read the ``git diff`` on ``testdata/reload_golden.json`` before committing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vnl_experiments.delays import evaluation
from vnl_experiments.delays.network_builders import _parse_net_params, get_architecture

HERE = Path(__file__).parent
CONFIG_DIR = HERE / "testdata" / "configs"
GOLDEN_PATH = HERE / "testdata" / "reload_golden.json"

#: Config fields holding a filesystem path. Recorded as basenames because the absolute
#: directory is machine-specific -- and because the basename *is* the thing the repair in
#: ``config_io`` decides: it keeps the run's file and swaps only the directory.
_PATH_FIELDS = ("walker_xml_path", "arena_xml_path", "reference_data_path")


def case_names() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.json"))


def _portable(value):
    """A JSON-comparable form of a config value, with paths reduced to basenames."""
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def rebuild(name: str) -> dict:
    """Reconstruct env config + net params from one stored ``config.json``."""
    payload = json.loads((CONFIG_DIR / f"{name}.json").read_text())
    env_params = payload["env_params"]
    net_params = payload.get("net_params", {})

    env_cls, default_fn = evaluation.resolve_env_class("", env_params, "Imitation")
    cfg = evaluation.parse_env_config(env_params, default_fn)

    env_dict = {k: _portable(v) for k, v in cfg.to_dict().items()}
    for field in _PATH_FIELDS:
        if field in env_dict:
            env_dict[field] = Path(str(env_dict[field])).name

    arch = get_architecture(net_params.get("network_class", ""))
    return {
        "env_class": env_cls.__name__,
        "env_config": env_dict,
        "net_params": _portable(_parse_net_params(net_params)),
        "architecture": arch.name if arch is not None else None,
    }


def load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text())


@pytest.mark.parametrize("name", case_names())
def test_reload_matches_golden(name: str) -> None:
    """Rebuilding this checkpoint's config yields exactly what it always has."""
    assert GOLDEN_PATH.exists(), (
        f"{GOLDEN_PATH} is missing. Generate it with "
        f"`python -m vnl_experiments.delays.reload_golden_test --regenerate`."
    )
    golden = load_golden()
    assert name in golden, (
        f"No golden entry for {name!r}. A new fixture needs regenerating -- and the "
        f"regenerated values need reading, not just committing."
    )
    assert rebuild(name) == golden[name]


def test_every_fixture_is_covered() -> None:
    """The golden file and the fixture directory describe the same set of cases.

    Guards the failure mode where a fixture is added but never generated, or deleted but
    left in the golden file -- either way the suite would silently stop testing it.
    """
    assert sorted(load_golden()) == case_names()


def test_legacy_string_values_survive_as_numbers() -> None:
    """The 2026-08-24 truncation bug, pinned directly.

    ``_parse_net_params`` must decode ``"0.01"`` to 0.01, not to ``int("0.01")``-style 0.
    The golden file would catch this too, but only as one diff line among many; this says
    what actually went wrong.
    """
    parsed = rebuild("legacy_stringified")["net_params"]
    for key in ("entropy_weight", "kl_weight", "min_std", "latent_min_std"):
        if key in parsed and isinstance(parsed[key], (int, float)):
            assert parsed[key] != 0, f"{key} was truncated to zero"


@pytest.mark.parametrize("name", case_names())
def test_video_config_is_the_shared_one_plus_its_overrides(name: str) -> None:
    """`eval_videos` must reconstruct the same env an eval does.

    It used to hold a second copy of the field whitelist. The two drifting is how a video
    ends up showing a different body, solver or reward shaping than the number beside it
    was measured on. Now it calls the shared reconstruction and overrides only the render
    specifics -- so everything *except* those specifics must still match exactly.
    """
    from vnl_experiments.delays import eval_videos

    payload = json.loads((CONFIG_DIR / f"{name}.json").read_text())
    env_params = payload["env_params"]
    _, default_fn = evaluation.resolve_env_class("", env_params, "Imitation")

    shared = evaluation.parse_env_config(env_params, default_fn).to_dict()
    video = eval_videos.parse_imitation_env_config(
        env_params, "assets/eval.h5", 1500, default_fn).to_dict()

    # The documented render-only overrides, and nothing else.
    expected_differences = {"reference_data_path", "clip_length", "naconmax",
                            "njmax", "clip_set"}
    differing = {k for k in set(shared) | set(video)
                 if str(shared.get(k)) != str(video.get(k))}
    assert differing <= expected_differences


def test_asset_paths_keep_the_run_s_own_file() -> None:
    """The run's XML basename survives the repair; only the directory is localised.

    This is the collision-model-xml failure: falling back to the local default silently
    swaps the *body*, and the eval then describes a different animal than the run trained.
    """
    payload = json.loads((CONFIG_DIR / "encdec_modern.json").read_text())
    trained = Path(str(payload["env_params"]["walker_xml_path"])).name
    rebuilt = rebuild("encdec_modern")["env_config"]["walker_xml_path"]
    assert rebuilt == trained


def _regenerate() -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    golden = {name: rebuild(name) for name in case_names()}
    GOLDEN_PATH.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_PATH} ({len(golden)} cases)")


if __name__ == "__main__":
    import sys

    if "--regenerate" in sys.argv:
        _regenerate()
    else:
        print(__doc__)
        print("Pass --regenerate to rewrite the golden file.")
