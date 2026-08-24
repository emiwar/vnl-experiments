"""Which code produced this: the git state of every repo that can change a number.

Three repos are installed as editable packages, and all three can move results:

* ``vnl_experiments`` — the training scripts, networks and eval code;
* ``nnx_ppo`` — the PPO algorithm itself;
* ``vnl_playground`` — the environment, its reward terms and its terminations.

WandB records only the commit of the repo the *script* lives in, so the other two were
invisible: an ``nnx-ppo`` change alters training and a ``vnl-playground`` change alters
the task, and neither left a trace in the run index or in any artifact sidecar.

``dirty`` is recorded alongside ``commit`` because these are editable checkouts and the
cluster working copy is known to drift from what is committed (see the "cluster working
copy drifts" trap in ``analysis/README.md``). A dirty tree means the commit does **not**
identify the code that ran, and an analysis spanning one should say so.

Discovery uses ``importlib.util.find_spec``, which locates a package without importing
it, so stamping an artifact never drags MuJoCo into the process.
"""

from __future__ import annotations

import subprocess
from importlib.util import find_spec
from pathlib import Path
from typing import Any

#: Import names of the repos whose code can change a result. Order is stable so the
#: stamp is comparable across runs.
PACKAGES = ("vnl_experiments", "nnx_ppo", "vnl_playground")


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(cwd), *args],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 - provenance must never break a run
        return None


def _package_dir(name: str) -> Path | None:
    try:
        spec = find_spec(name)
    except Exception:  # noqa: BLE001
        return None
    origin = getattr(spec, "origin", None) if spec else None
    if not origin:
        locations = list(getattr(spec, "submodule_search_locations", None) or [])
        return Path(locations[0]) if locations else None
    return Path(origin).parent


def repo_state(path: Path) -> dict[str, Any] | None:
    """``{commit, dirty}`` for the repo containing ``path``, or None if not a checkout."""
    commit = _git(["rev-parse", "HEAD"], path)
    if not commit:
        return None
    status = _git(["status", "--porcelain"], path)
    return {"commit": commit, "dirty": bool(status)}


def repo_versions() -> dict[str, dict[str, Any]]:
    """Git state of each installed repo, keyed by import name.

    Missing or non-git packages are omitted rather than recorded as null, so an absent
    key means "not determinable", not "clean".
    """
    out: dict[str, dict[str, Any]] = {}
    for name in PACKAGES:
        directory = _package_dir(name)
        if directory is None:
            continue
        state = repo_state(directory)
        if state is not None:
            out[name] = state
    return out
