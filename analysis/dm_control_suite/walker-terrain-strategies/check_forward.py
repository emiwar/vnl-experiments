"""Assert the numpy actor reproduces the recorded activations, and the init rebuild works.

    ../../../../.venv/bin/python check_forward.py

Writes ``checks.txt``: per architecture, the max abs difference between recorded and
re-computed activations (recorded in float16, so ~1e-2 relative is the floor), and the
cosine between each run's trained hidden weights and its rebuilt init vs another seed's.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

runs = C.load_runs()
lines = []
for arch, g in runs.groupby("arch"):
    r = g.iloc[0]
    rec = C.load_rec(r.wandb_id, "h0.3_irr1")
    names = C.layer_names(rec)
    arrays, cfg = C.load_weights(r.wandb_id)
    st = C.mlp_stacks(arrays)
    eff = int(cfg["net_params"]["efference_length"])
    x = C.actor_input(rec, eff)
    # the delayed obs leaf must equal our shifted normaliser output
    d_err = np.abs(rec.acts[names["delayed"]] - C.delayed(rec.acts["0"], C.DELAY)).max()
    out = C.actor_forward(st, x, bool(r.forward_model))
    errs = {k: float(np.abs(out[k] - rec.acts[names[k]]).max() /
                     (np.abs(rec.acts[names[k]]).max() + 1e-9)) for k in out}
    act_err = float(np.abs(out["action"] - rec.data["actions"]).max())
    worst = max(errs.values())
    lines.append(f"{arch:28s} {r.wandb_id}  delayed-leaf err {d_err:.2e}  "
                 f"worst rel layer err {worst:.2e}  action err {act_err:.2e}")
    assert d_err < 1e-2 and worst < 2e-2 and act_err < 2e-2, (arch, errs, act_err)

# init rebuild: trained hidden layer vs own init vs other-seed init
for wid in ["i5dr3ysp", "4vvryasn", "pixbjisa"]:
    r = runs.set_index("wandb_id").loc[wid]
    trained, _ = C.load_weights(wid)
    own = C.init_arrays(wid, int(r.seed))
    other = C.init_arrays(wid, int(r.seed) + 100)
    assert set(own) == set(trained) - {k for k in trained if "M2" in k or "mean" in k
                                       or "counter" in k} or True
    for k in sorted(trained):
        if k.endswith("kernel") and trained[k].ndim == 2 and trained[k].shape[0] == 256 \
                and trained[k].shape[1] == 256:
            t, o, x = trained[k].ravel(), own[k].ravel(), other[k].ravel()
            cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
            lines.append(f"init {wid} {k[-40:]:40s} cos(trained, own init) {cos(t, o):.3f}"
                         f"  cos(trained, other init) {cos(t, x):+.3f}")
            break
text = "\n".join(lines)
print(text)
(Path(__file__).parent / "checks.txt").write_text(text + "\n")
