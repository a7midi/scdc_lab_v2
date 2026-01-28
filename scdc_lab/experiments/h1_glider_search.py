from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..graphs import (
    er_directed_multigraph,
    inject_knot,
    layered_random_dag,
    compute_condensation,
)
from ..world import WorldInstance, ThresholdRule, XorRule, random_lookup_rule
from ..schedule import simulate
from ..pockets import compute_essential_inputs, pockets_from_active_set
from ..scdc import SCDCConfig


def _node_depths(world: WorldInstance) -> Dict[int, int]:
    """Depth label per node derived from SCC condensation depth."""
    cond = compute_condensation(world.G)
    d_scc = cond.depth
    node_to_scc = cond.node_to_scc
    return {v: int(d_scc.get(node_to_scc[v], 0)) for v in world.G.nodes()}


def _centroid_depth(nodes: Set[int], depth: Dict[int, int]) -> float:
    if not nodes:
        return float("nan")
    return float(np.mean([depth.get(v, 0) for v in nodes]))


def _jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))


def _pick_knot_nodes_layered(
    n_layers: int,
    layer_size: int,
    k: int,
    rng: np.random.Generator,
    knot_layer: Optional[int] = None,
) -> List[int]:
    if knot_layer is None:
        # Avoid extreme boundary layers by default
        lo = 1 if n_layers >= 3 else 0
        hi = n_layers - 2 if n_layers >= 3 else n_layers - 1
        knot_layer = int(rng.integers(lo, hi + 1))
    knot_layer = int(max(0, min(n_layers - 1, knot_layer)))
    base = knot_layer * layer_size
    candidates = np.arange(base, base + layer_size, dtype=int)
    if k > layer_size:
        k = layer_size
    return [int(x) for x in rng.choice(candidates, size=k, replace=False)]


def build_world(
    graph_type: str,
    n: int,
    p: float,
    layers: int,
    p_forward: float,
    p_skip: float,
    knot_k: int,
    seed: int,
    rule: str,
    threshold: int,
    alphabet_k: int,
    knot_layer: Optional[int],
    knot_internal_p: float,
    knot_parallel: int,
    add_back_edges: bool,
) -> Tuple[WorldInstance, List[int]]:
    rng = np.random.default_rng(seed)

    if graph_type == "er":
        G = er_directed_multigraph(n=n, p=p, seed=seed, allow_self_loops=False, max_parallel=1)
        knot_nodes = [int(x) for x in rng.choice(np.arange(n, dtype=int), size=min(knot_k, n), replace=False)]
    elif graph_type == "layered":
        if layers <= 0:
            raise ValueError("--layers must be >= 1")
        if n % layers != 0:
            raise ValueError(f"For layered graphs require n divisible by layers. Got n={n}, layers={layers}.")
        layer_size = n // layers
        G = layered_random_dag(n_layers=layers, layer_size=layer_size, p_forward=p_forward, p_skip=p_skip, seed=seed)
        knot_nodes = _pick_knot_nodes_layered(layers, layer_size, knot_k, rng, knot_layer=knot_layer)
    else:
        raise ValueError("--graph_type must be one of: er, layered")

    # Inject a dense local "knot" (defect seed)
    G = inject_knot(
        G,
        knot_nodes=knot_nodes,
        p_internal=knot_internal_p,
        extra_parallel=max(1, int(knot_parallel)),
        add_back_edges=bool(add_back_edges),
        seed=seed + 1,
    )

    # Choose local rule family
    def rule_factory(v: int, in_sizes: List[int], rrng: np.random.Generator):
        if rule == "threshold":
            return ThresholdRule(threshold=int(threshold), vacuum_zero=True)
        if rule == "xor":
            return XorRule()
        if rule == "random":
            return random_lookup_rule(in_sizes=in_sizes, out_size=int(alphabet_k), rng=rrng, vacuum_fixed=0)
        raise ValueError("--rule must be one of: threshold, xor, random")

    world = WorldInstance.homogeneous(G=G, k=int(alphabet_k), rule_factory=rule_factory, seed=seed + 2)
    return world, knot_nodes


def main() -> None:
    ap = argparse.ArgumentParser(description="H1: Search for propagating autonomous pockets (glider-like matter).")

    ap.add_argument("--graph_type", choices=["er", "layered"], default="er")
    ap.add_argument("--n", type=int, default=240, help="Number of nodes (or total nodes for layered).")
    ap.add_argument("--p", type=float, default=0.02, help="ER edge probability (er only).")

    ap.add_argument("--layers", type=int, default=12, help="Number of layers (layered only).")
    ap.add_argument("--p_forward", type=float, default=0.06, help="Forward-edge probability (layered only).")
    ap.add_argument("--p_skip", type=float, default=0.01, help="Skip-edge probability (layered only).")
    ap.add_argument("--knot_layer", type=int, default=None, help="Layer index to place knot in (layered only).")

    ap.add_argument("--knot_k", type=int, default=12, help="Number of nodes in the injected knot.")
    ap.add_argument("--knot_internal_p", type=float, default=0.8)
    ap.add_argument("--knot_parallel", type=int, default=2)
    ap.add_argument("--no_back_edges", action="store_true", help="Disable reverse edges inside knot.")

    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--schedule_per_tick", action="store_true", help="Resample schedule each tick (default).")
    ap.add_argument("--fixed_schedule", action="store_true", help="Use one fixed schedule across all ticks.")

    ap.add_argument("--rule", choices=["threshold", "xor", "random"], default="threshold")
    ap.add_argument("--threshold", type=int, default=2)
    ap.add_argument("--alphabet_k", type=int, default=2)

    ap.add_argument("--ess_samples", type=int, default=2048, help="Samples for essential-input estimation.")
    ap.add_argument("--ess_enum_cap", type=int, default=2048)

    ap.add_argument("--out_csv", type=str, default="h1.csv")
    args = ap.parse_args()

    schedule_per_tick = True
    if args.fixed_schedule:
        schedule_per_tick = False
    elif args.schedule_per_tick:
        schedule_per_tick = True

    world, knot_nodes = build_world(
        graph_type=args.graph_type,
        n=args.n,
        p=args.p,
        layers=args.layers,
        p_forward=args.p_forward,
        p_skip=args.p_skip,
        knot_k=args.knot_k,
        seed=args.seed,
        rule=args.rule,
        threshold=args.threshold,
        alphabet_k=args.alphabet_k,
        knot_layer=args.knot_layer,
        knot_internal_p=args.knot_internal_p,
        knot_parallel=args.knot_parallel,
        add_back_edges=(not args.no_back_edges),
    )

    print(f"Knot nodes: {knot_nodes}")

    # Initial state: vacuum except knot nodes set to 1
    x0 = world.vacuum_state(0)
    for v in knot_nodes:
        x0[v] = 1 if args.alphabet_k <= 2 else int(1 % args.alphabet_k)

    # Simulate
    states = simulate(world, x0, steps=args.steps, seed=args.seed + 3, schedule_per_tick=schedule_per_tick)

    # Essential inputs
    cfg = SCDCConfig(
        max_enumerate_tuples_per_vertex=int(args.ess_enum_cap),
        sample_tuples_per_vertex=int(args.ess_samples),
        seed=int(args.seed + 4),
    )
    ess = compute_essential_inputs(world, cfg=cfg)

    depth = _node_depths(world)

    rows: List[Dict[str, float]] = []
    prev_pocket: Set[int] = set()

    for t, x in enumerate(states):
        active = {v for v, val in x.items() if int(val) != 0}
        pockets = pockets_from_active_set(world, active, ess)

        # Choose the largest pocket (if any)
        pocket: Set[int] = set()
        if pockets:
            pocket = max(pockets, key=lambda P: len(P))

        row = {
            "t": int(t),
            "active_size": int(len(active)),
            "active_centroid_depth": _centroid_depth(active, depth),
            "pocket_size": int(len(pocket)),
            "pocket_centroid_depth": _centroid_depth(pocket, depth),
            "jaccard_prev": (_jaccard(pocket, prev_pocket) if t > 0 else float("nan")),
            "active_jaccard_prev": (_jaccard(active, {v for v, val in states[t-1].items() if int(val) != 0}) if t > 0 else float("nan")),
        }
        rows.append(row)
        prev_pocket = set(pocket)

    # Write CSV
    fieldnames = list(rows[0].keys()) if rows else ["t", "active_size", "pocket_size"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {args.out_csv}")

    # Quick summary
    pocket_sizes = [int(r["pocket_size"]) for r in rows]
    nonzero = [s for s in pocket_sizes if s > 0]
    if nonzero:
        print(f"Pocket size: min={min(nonzero)} median={int(np.median(nonzero))} max={max(nonzero)} (nonzero steps={len(nonzero)}/{len(rows)})")
    else:
        print("No pockets detected (all pocket_size=0).")

    # Drift proxy: pocket centroid change ignoring nans
    cd = [r["pocket_centroid_depth"] for r in rows if np.isfinite(r["pocket_centroid_depth"])]
    if len(cd) >= 2:
        print(f"Pocket centroid drift: Δdepth ≈ {cd[-1] - cd[0]:.3f} over {len(cd)-1} ticks")


if __name__ == "__main__":
    main()
