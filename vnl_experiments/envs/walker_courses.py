"""Fixed walker-terrain courses: one block layout, generated once and committed.

Why these exist: during training ``WalkerTerrain`` draws a fresh layout per episode and
draws every recycled block from the rng split of the step it is recycled on, so the ground a
walker meets depends on its own trajectory. Two policies -- or one policy at two speeds --
given the same seed walk different terrain. A *course* is a long, precomputed list of blocks
(absolute centre x and top height); ``WalkerTerrain.reset_on_course`` places blocks from it
by position, so every policy walks the same ground. See the env's module docstring.

Courses are drawn by exactly the formula training uses (``walker_terrain.sample_blocks``)
and placed exactly as ``_initial_mocap_pos`` places the first ``K`` blocks, so a course from
the training settings is a sample of the training distribution. Each is 400 m long:
an episode is 25 s, the walker's physical speed ceiling is ~10 m/s, so the worst case is
~260 m including the run-up and the queue ahead of the torso, and 400 m leaves ~1.5x
margin. A course that runs out anyway parks blocks at ``PARK_X``, which the recorder
treats as an error.

Courses live in ``terrain_courses/`` as ``<name>.npz``, with ``courses.json`` holding
the recipe for each one and the *sets* the recorder uses (a set lists one course per
episode, cycled). Regenerate deterministically with::

    python -m vnl_experiments.envs.walker_courses make
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

COURSES_DIR = Path(__file__).resolve().parent / "terrain_courses"
CATALOG_PATH = COURSES_DIR / "courses.json"

#: Course length in metres (see the module docstring for why 400).
DEFAULT_LENGTH_M = 400.0


@dataclass(frozen=True)
class Course:
    """One fixed course: block centres ``x`` (ascending) and block ``top`` heights."""

    name: str
    x: np.ndarray
    top: np.ndarray
    meta: dict[str, Any]

    @property
    def n_blocks(self) -> int:
        return int(self.x.shape[0])

    @property
    def sha256(self) -> str:
        """Hash of the block arrays (not the metadata): the course's identity."""
        digest = hashlib.sha256()
        for arr in (self.x, self.top):
            a = np.ascontiguousarray(arr, dtype=np.float32)
            digest.update(str(a.shape).encode())
            digest.update(a.tobytes())
        return digest.hexdigest()

    def save(self, path: Path | str) -> None:
        np.savez(path, x=self.x.astype(np.float32), top=self.top.astype(np.float32),
                 meta=np.array(json.dumps(self.meta, sort_keys=True)))

    @classmethod
    def load(cls, path: Path | str, name: str | None = None) -> "Course":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            return cls(name=name or path.stem, x=data["x"], top=data["top"],
                       meta=json.loads(str(data["meta"])))


def terrain_config(**overrides: Any):
    """The default ``terrain`` sub-config with ``overrides`` applied."""
    from vnl_experiments.envs.walker_terrain import default_terrain_config

    cfg = default_terrain_config()
    for key, value in overrides.items():
        if key not in cfg:
            raise KeyError(f"unknown terrain key {key!r}")
        cfg[key] = type(cfg[key])(value)
    return cfg


def make_course(terrain, *, seed: int, name: str,
                length_m: float = DEFAULT_LENGTH_M) -> Course:
    """Draw a course from ``terrain`` settings, placed as the env places its blocks."""
    import jax

    from vnl_experiments.envs.walker_terrain import sample_blocks

    L = float(terrain.block_length)
    sigma = float(terrain.irregularity)
    min_period = 2 * L + float(terrain.gap) * (1 - sigma)
    n = int(math.ceil((length_m - float(terrain.start_gap)) / min_period)) + 2
    heights, periods = sample_blocks(terrain, jax.random.key(seed), n)
    heights, periods = np.asarray(heights, np.float32), np.asarray(periods, np.float32)
    first_centre = float(terrain.start_gap) + L
    x = (first_centre + np.cumsum(periods) - periods[0]).astype(np.float32)
    keep = x - L <= length_m          # every block that starts within the course length
    meta = {
        "name": name, "seed": int(seed), "length_m": float(length_m),
        **{k: float(terrain[k]) for k in ("height", "gap", "irregularity", "block_length",
                                          "block_halfheight", "start_gap")},
    }
    return Course(name=name, x=x[keep], top=heights[keep], meta=meta)


# --------------------------------------------------------------------------------------
# catalog: recipes and sets
# --------------------------------------------------------------------------------------


def catalog() -> dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text())


def load_course(name: str) -> Course:
    path = COURSES_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"no course {name!r} in {COURSES_DIR}; known: "
                                f"{sorted(p.stem for p in COURSES_DIR.glob('*.npz'))}")
    return Course.load(path, name)


def course_set(set_name: str) -> list[str]:
    """The course names a set cycles through, one per episode."""
    sets = catalog()["sets"]
    if set_name not in sets:
        raise KeyError(f"unknown course set {set_name!r}; known: {sorted(sets)}")
    return list(sets[set_name])


def courses_for_episodes(set_name: str, n_episodes: int) -> list[Course]:
    """Episode ``i`` walks course ``i mod len(set)``."""
    names = course_set(set_name)
    return [load_course(names[i % len(names)]) for i in range(n_episodes)]


def set_sha256(set_name: str) -> str:
    """Identity of a set's contents: its course names and their block arrays, in order."""
    digest = hashlib.sha256()
    for name in course_set(set_name):
        digest.update(name.encode())
        digest.update(load_course(name).sha256.encode())
    return digest.hexdigest()


def stack_courses(courses: Sequence[Course]) -> tuple[np.ndarray, np.ndarray]:
    """``[N, M]`` arrays for a vmapped batch, padded with ``x = +inf`` (never reached)."""
    m = max(c.n_blocks for c in courses)
    xs = np.full((len(courses), m), np.inf, np.float32)
    tops = np.zeros((len(courses), m), np.float32)
    for i, c in enumerate(courses):
        xs[i, : c.n_blocks] = c.x
        tops[i, : c.n_blocks] = c.top
    return xs, tops


def regenerate(out_dir: Path = COURSES_DIR) -> list[Course]:
    """Rebuild every course from its recipe in ``courses.json``."""
    made = []
    for name, recipe in catalog()["courses"].items():
        recipe = dict(recipe)
        seed = int(recipe.pop("seed"))
        length = float(recipe.pop("length_m", DEFAULT_LENGTH_M))
        course = make_course(terrain_config(**recipe), seed=seed, name=name,
                             length_m=length)
        course.save(out_dir / f"{name}.npz")
        made.append(course)
    return made


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vnl_experiments.envs.walker_courses")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("make", help="regenerate every course from courses.json")
    sub.add_parser("ls", help="list courses and sets")
    args = parser.parse_args(argv)
    if args.command == "make":
        for c in regenerate():
            print(f"{c.name}: {c.n_blocks} blocks to x={c.x[-1]:.1f} m, "
                  f"tops {c.top.min():.3f}-{c.top.max():.3f} m, sha {c.sha256[:12]}")
    else:
        cat = catalog()
        for name in cat["courses"]:
            c = load_course(name)
            print(f"{name}: {c.n_blocks} blocks, sha {c.sha256[:12]}, {c.meta}")
        for set_name, names in cat["sets"].items():
            print(f"set {set_name}: {names}  (sha {set_sha256(set_name)[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
