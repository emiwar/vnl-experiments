"""Does ``delay_k`` reach the network at all when the decoder has no proprioception?

Q2b of this folder ("does the delay length matter when there is no proprioceptive
input?") is not really an empirical question: ``build_delay_network`` puts the ``Delay``
layer *inside the proprioception branch*, and ``dec_use_proprioception=False`` does not
construct that branch. So ``delay_k`` should be dead code for those runs.

"Should be" is what this script removes. It builds the enc-dec network twice from the
recorded ``net_params`` of the real runs -- once with each run's own ``delay_k`` -- and
checks that with the proprioception branch ablated the two networks are **identical**:
same parameter count, same module tree, and bit-identical actions on the same input with
the same seed. It then does the same with proprioception present, where the two must
*differ*, so that a vacuous pass (e.g. a stub that ignores everything) cannot look like a
success.

No env, no checkpoint, no GPU: ``build_delay_network`` reads only
``non_flattened_observation_size`` and ``action_size`` off the env, so a stub carrying the
real rodent's sizes is enough, and the question is about graph structure rather than
trained weights.

    ../.venv/bin/python analysis/position-control-open-loop/check_delay_inert.py
    ../.venv/bin/python analysis/position-control-open-loop/check_delay_inert.py --check

Writes ``delay_inert.txt``; ``--check`` diffs against the committed copy instead.
"""

import argparse
import difflib
from pathlib import Path

import jax
import numpy as np
from flax import nnx

from vnl_experiments.delays.network_builders import build_delay_network

HERE = Path(__file__).resolve().parent
OUT = HERE / "delay_inert.txt"

#: The real widths of the two observation groups, and the action width. Only the totals
#: reach `build_delay_network` (it sums each group's subtree), and they are not guessed:
#: they are solved from the parameter counts these runs recorded in
#: `final_eval/params/{encoder,decoder,critic}`, and check D below asserts the round trip.
TASK_OBS_SIZE = 640      # 5 reference frames of root / quat / joint / body targets
PROPRIO_SIZE = 277
ACTION_SIZE = 38

#: `final_eval/params/total` as the runs themselves reported it, keyed by
#: (dec_use_proprioception, dec_use_intention, efference_length). If the stub's widths were
#: wrong, every parameter count printed here would be wrong in a way nothing else would
#: catch -- so D checks them against the runs rather than against arithmetic done here.
RECORDED_TOTALS = {
    (False, True, 0): 3_983_501,    # dkglvuw8 / wbyipflf / 23267t68
    (False, True, 5): 4_080_781,    # mbjli503
    (False, True, 10): 4_178_061,   # 3gw1ndwj
    (True, True, 0): 4_125_325,     # 16jfo5vu (delay 0) -- also n4amxgv0 (delay 10, eff 0)
    (True, True, 10): 4_319_885,    # jtl5r0px
    (True, False, 10): 3_154_509,   # 9hyirkcx (nointent)
}

#: The net_config every one of these runs shares.
NET_PARAMS = {
    "enc_hidden_sizes": [512] * 4,
    "dec_hidden_sizes": [512] * 4,
    "critic_hidden_sizes": [1024, 1024],
    "latent_size": 32,
    "kl_weight": 0.001,
    "latent_min_std": 0.01,
    "entropy_weight": 0.01,
    "min_std": 0.1,
    "std_scale": 1.0,
    "activation": "swish",
    "normalize_obs": True,
    "initializer_scale": 1.0,
}

#: (delay_k, efference_length) of the no-proprioception runs in `runs.csv`. The point of
#: the pairing is that these are the *labels* the run names carry -- `delay5_eff0_noproprio`
#: against `delay0_eff0_noproprio` -- and the claim is that the first two collapse onto one
#: network.
NOPROPRIO_RUNS = [
    ("dkglvuw8", 0, 0),
    ("wbyipflf", 0, 0),
    ("23267t68", 5, 0),
    ("mbjli503", 5, 5),
    ("3gw1ndwj", 10, 10),
]


class StubEnv:
    """Just the two attributes ``build_delay_network`` reads off an env."""

    non_flattened_observation_size = {
        "state": {"task_obs": TASK_OBS_SIZE, "proprioception": PROPRIO_SIZE}
    }
    action_size = ACTION_SIZE


def build(*, delay_k: int, efference_length: int, use_proprio: bool,
          use_intention: bool = True, seed: int = 42):
    params = {
        **NET_PARAMS,
        "delay_k": delay_k,
        "efference_length": efference_length,
        "dec_use_intention": use_intention,
        "dec_use_proprioception": use_proprio,
    }
    return build_delay_network(params, StubEnv(), nnx.Rngs(seed))


def observation(seed: int = 0):
    task_key, proprio_key = jax.random.split(jax.random.key(seed))
    return {"state": {
        "task_obs": jax.random.normal(task_key, (1, TASK_OBS_SIZE)),
        "proprioception": jax.random.normal(proprio_key, (1, PROPRIO_SIZE)),
    }}


def forward(net, obs):
    """One forward pass, reduced to the quantities that are a function of weights + input.

    Not the sampled action: the sampler draws from an RNG stream, so two draws agreeing
    would confound "same network" with "same random numbers". ``mu``/``sigma`` (the policy
    and bottleneck distribution parameters, reported in ``metrics``) and the critic's
    value estimate are deterministic, which is what a structural claim needs.
    """
    out = net(net.initialize_state(1), obs)
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(out.metrics)]
    leaves.append(np.asarray(out.output.value_estimates))
    return leaves


def tree_shape(tree) -> str:
    """The pytree's paths and shapes as text -- structure without values."""
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(sorted(f"{jax.tree_util.keystr(path)} {np.shape(leaf)}"
                            for path, leaf in flat))


def param_count(net) -> int:
    state = nnx.state(net, nnx.Param)
    return int(sum(np.prod(leaf.shape) for leaf in jax.tree.leaves(state)))


def compare(delay_a: int, delay_b: int, *, efference_length: int, use_proprio: bool):
    """Build the same net at two delays and report every way they could differ."""
    net_a = build(delay_k=delay_a, efference_length=efference_length,
                  use_proprio=use_proprio)
    net_b = build(delay_k=delay_b, efference_length=efference_length,
                  use_proprio=use_proprio)
    obs = observation()
    same_params = param_count(net_a) == param_count(net_b)
    same_weight_tree = (tree_shape(nnx.state(net_a, nnx.Param))
                        == tree_shape(nnx.state(net_b, nnx.Param)))
    # The carry state is where a Delay shows up structurally: it owns the ring buffer of
    # past observations, so a delayed branch has a state leaf a delay-0 branch does not.
    same_carry_tree = (tree_shape(net_a.initialize_state(1))
                       == tree_shape(net_b.initialize_state(1)))
    out_a, out_b = forward(net_a, obs), forward(net_b, obs)
    max_abs = max(float(np.abs(x - y).max()) for x, y in zip(out_a, out_b))
    return {
        "params": (param_count(net_a), param_count(net_b)),
        "same_params": same_params,
        "same_weight_tree": same_weight_tree,
        "same_carry_tree": same_carry_tree,
        "max_abs_output_diff": max_abs,
        "identical": (same_params and same_weight_tree and same_carry_tree
                      and max_abs == 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="diff against the committed delay_inert.txt")
    args = parser.parse_args()

    lines = ["Is delay_k inert without the proprioception branch?", ""]
    failures = []

    lines.append("A. dec_use_proprioception=False -- the two networks must be IDENTICAL")
    for efference_length in (0, 5, 10):
        r = compare(0, 10, efference_length=efference_length, use_proprio=False)
        lines.append(_line(efference_length, r))
        if not r["identical"]:
            failures.append(f"noproprio eff={efference_length}: not identical")
    lines.append("")

    lines.append("B. dec_use_proprioception=True -- the two networks must DIFFER, "
                 "so that A cannot pass vacuously")
    for efference_length in (0, 5, 10):
        r = compare(0, 10, efference_length=efference_length, use_proprio=True)
        lines.append(_line(efference_length, r))
        if r["identical"]:
            failures.append(f"proprio eff={efference_length}: did not differ")
    lines.append("")
    lines.append("   (B's weight tree and parameter count *are* the same -- a Delay layer")
    lines.append("    has no parameters. What separates the two is the carry state, which")
    lines.append("    holds the delay buffer, and hence the output.)")
    lines.append("")

    lines.append("C. the no-proprioception runs in runs.csv, grouped by the network they")
    lines.append("   actually built")
    groups: dict[int, list[str]] = {}
    for wandb_id, delay_k, efference_length in NOPROPRIO_RUNS:
        groups.setdefault(efference_length, []).append(
            f"{wandb_id} (delay{delay_k}_eff{efference_length})")
    for efference_length, members in sorted(groups.items()):
        net = build(delay_k=0, efference_length=efference_length, use_proprio=False)
        lines.append(f"   eff={efference_length:<3d} {param_count(net):>10,d} params  "
                     f"{', '.join(members)}")
    lines.append("")
    lines.append("   So delay0_eff0_noproprio and delay5_eff0_noproprio are one and the")
    lines.append("   same architecture: their reward difference is a replicate difference,")
    lines.append("   not a delay effect. What separates the rows above is efference_length.")
    lines.append("")

    lines.append("D. the stub's observation widths, checked against the parameter counts")
    lines.append("   the runs themselves recorded in final_eval/params/total")
    for (use_proprio, use_intention, eff), recorded in sorted(RECORDED_TOTALS.items()):
        net = build(delay_k=0, efference_length=eff, use_proprio=use_proprio,
                    use_intention=use_intention)
        built = param_count(net)
        ok = built == recorded
        lines.append(f"   proprio={str(use_proprio):<5s} intention={str(use_intention):<5s} "
                     f"eff={eff:<3d} built {built:>10,d}  recorded {recorded:>10,d}  "
                     f"-> {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"param count mismatch for "
                            f"(proprio={use_proprio}, intention={use_intention}, eff={eff})")
    lines.append("")

    lines.append("FAILURES: " + (", ".join(failures) if failures else "none"))
    text = "\n".join(lines) + "\n"

    if args.check:
        if not OUT.exists():
            print(f"CHECK: {OUT.name} does not exist")
            raise SystemExit(1)
        committed = OUT.read_text()
        if committed == text:
            print(f"CHECK: {OUT.name} unchanged")
        else:
            print("\n".join(difflib.unified_diff(
                committed.splitlines(), text.splitlines(),
                fromfile=f"committed/{OUT.name}", tofile=f"rebuilt/{OUT.name}",
                lineterm="")))
            raise SystemExit(1)
    else:
        OUT.write_text(text)
        print(text)

    if failures:
        raise SystemExit("the inertness claim does not hold; see above")


def _line(efference_length: int, r: dict) -> str:
    return (f"   eff={efference_length:<3d} delay 0 vs 10: "
            f"params {r['params'][0]:,} vs {r['params'][1]:,}, "
            f"weights={_yn(r['same_weight_tree'])} carry={_yn(r['same_carry_tree'])} "
            f"max|d out|={r['max_abs_output_diff']:.3e}"
            f"  -> {'IDENTICAL' if r['identical'] else 'DIFFERENT'}")


def _yn(same: bool) -> str:
    return "same" if same else "diff"


if __name__ == "__main__":
    main()
