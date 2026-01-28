from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ..graphs import er_directed_multigraph, layered_random_dag, inject_knot, compute_condensation
from ..world import WorldInstance, make_rule_factory
from ..scdc import SCDCConfig, compute_lambda_star, quotient_loss, quotient_world, quotient_state
from ..schedule import simulate
from ..pockets import compute_essential_inputs, pockets_from_active_set
from ..pocket_activity import active_set_from_state
from ..geometry import (
    measure_light_cone_growth,
    curvature_field_on_condensation,
    embed_condensation_mds,
    shortest_path_geodesic,
    lensing_score_for_path,
    geodesic_deviation_overlap,
)
from ..genesis import (
    motif_energy,
    ConsistencyEnergyConfig,
    consistency_energy,
    AnnealConfig,
    run_genesis,
)


def _spectral_gap_sym(G: nx.MultiDiGraph, k: int = 5) -> float:
    """Return a crude spectral gap proxy of the symmetrized adjacency."""
    import scipy.sparse.linalg as spla
    from scipy import sparse

    nodes = list(G.nodes())
    if len(nodes) < 3:
        return 0.0
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=float)
    As = (A + A.T) * 0.5
    try:
        vals = spla.eigsh(As, k=min(k, As.shape[0] - 1), return_eigenvectors=False)
        vals = np.sort(vals)[::-1]
        if len(vals) < 2:
            return 0.0
        return float(vals[0] - vals[1])
    except Exception:
        return float("nan")


def plot_spacetime_embedding(
    G: nx.MultiDiGraph,
    out_png: str,
    title: str = "Emergent Spacetime (Condensation MDS)",
) -> None:
    cond = compute_condensation(G)
    nodes, X = embed_condensation_mds(G)
    idx = {n: i for i, n in enumerate(nodes)}

# --- PATCH START: Handle 1D Collapse ---
    # If the universe is a perfect line/tube, MDS might return shape (N, 1)
    if len(X.shape) == 2 and X.shape[1] == 1:
        # Pad with a column of zeros to make it 2D for scatter plot
        X = np.hstack([X, np.zeros((X.shape[0], 1))])
    # --- PATCH END ---

    depths = np.array([cond.depth[n] for n in nodes], dtype=float)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=depths, s=15)
    plt.title(title)
    plt.colorbar(label="Depth (time)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_geodesic_lensing(
    G: nx.MultiDiGraph,
    matter_scc: Set[int],
    out_png: str,
    n_pairs: int = 25,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(int(seed))
    cond = compute_condensation(G)
    dag = cond.dag
    nodes, X = embed_condensation_mds(G)
    idx = {n: i for i, n in enumerate(nodes)}

    # pick random endpoints, require connectivity
    UG = dag.to_undirected()
    comps = list(nx.connected_components(UG))
    if not comps:
        return {"n_paths": 0, "mean_min_dist": float("nan"), "mean_mean_dist": float("nan")}
    largest = max(comps, key=len)
    candidates = sorted(int(x) for x in largest)

    # Guard: if the condensation is trivial (<=1 node in the largest connected component),
    # we cannot sample endpoint pairs for geodesics.
    if len(candidates) < 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], s=12)
        if matter_scc:
            pts = np.array([X[idx[m]] for m in matter_scc if m in idx])
            if len(pts) > 0:
                plt.scatter(pts[:, 0], pts[:, 1], s=40, marker="x")
        plt.title("Geodesics (insufficient endpoints: condensation too small)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        return {"n_paths": 0, "mean_min_dist": float("nan"), "mean_mean_dist": float("nan")}


    paths: List[List[int]] = []
    min_dists: List[int] = []
    mean_dists: List[float] = []

    for _ in range(int(n_pairs)):
        s, t = rng.choice(candidates, size=2, replace=False)
        path = shortest_path_geodesic(dag, int(s), int(t), undirected=True)
        if path is None:
            continue
        dmin, dmean = lensing_score_for_path(path, set(matter_scc), dag)
        paths.append(path)
        min_dists.append(int(dmin))
        mean_dists.append(float(dmean))

    # plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], s=12)
    # matter in red-ish (use default colormap by re-scattering)
    if matter_scc:
        pts = np.array([X[idx[m]] for m in matter_scc if m in idx])
        if len(pts) > 0:
            plt.scatter(pts[:, 0], pts[:, 1], s=40, marker="x")

    # overlay a few paths
    for path in paths[: min(10, len(paths))]:
        pts = np.array([X[idx[p]] for p in path if p in idx])
        if len(pts) >= 2:
            plt.plot(pts[:, 0], pts[:, 1], linewidth=1)

    plt.title("Geodesics (shortest paths) and Matter Region")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    return {
        "n_paths": int(len(paths)),
        "mean_min_dist": float(np.mean(min_dists)) if min_dists else float("nan"),
        "mean_mean_dist": float(np.mean(mean_dists)) if mean_dists else float("nan"),
    }


def run_pipeline(args: argparse.Namespace) -> Dict:
    rng = np.random.default_rng(int(args.seed))

    # 1) Initial graph
    if args.graph_type == "er":
        G0 = er_directed_multigraph(n=int(args.n), p=float(args.p), seed=int(args.seed), max_parallel=int(args.max_parallel))
    elif args.graph_type == "layered":
        # n interpreted as total nodes
        n_layers = int(args.layers)
        layer_size = int(args.n // max(1, n_layers))
        G0 = layered_random_dag(
            n_layers=n_layers,
            layer_size=layer_size,
            p_forward=float(args.p_forward),
            p_skip2=float(args.p_skip2),
            p_skip3=float(args.p_skip3),
            seed=int(args.seed),
        )
    else:
        raise ValueError(f"Unknown graph_type: {args.graph_type}")

    print(f"Initial graph: N={G0.number_of_nodes()} E={G0.number_of_edges()}")

    # 2) Genesis (optional)
    G = G0
    genesis_hist: List[float] = []

    if args.genesis_steps > 0:
        if args.energy_mode == "motif":
            energy_fn = lambda H: motif_energy(H, w_d=float(args.w_d), w_t=float(args.w_t))
        elif args.energy_mode == "scdc":
            scdc_cfg = SCDCConfig(
                seed=int(args.seed),
                max_iterations=int(args.scdc_iters),
                sample_tuples_per_vertex=int(args.scdc_samples),
                max_diamond_state_samples=int(args.scdc_diamond_samples),
            )
            e_cfg = ConsistencyEnergyConfig(
                alphabet_k=int(args.k),
                rule=str(args.rule),
                threshold=int(args.threshold),
                vacuum_fixed=int(args.vacuum),
                scdc=scdc_cfg,
            )
            energy_fn = lambda H: consistency_energy(H, e_cfg)
        else:
            raise ValueError(f"Unknown energy_mode: {args.energy_mode}")

        a_cfg = AnnealConfig(
            steps=int(args.genesis_steps),
            temp_start=float(args.temp_start),
            cooling_rate=float(args.cooling_rate),
            seed=int(args.seed),
        )
        print(f"--- GENESIS ({args.energy_mode}) START ---")
        G, genesis_hist = run_genesis(G0, energy_fn=energy_fn, cfg=a_cfg, progress_every=int(args.progress_every))
        print(f"--- GENESIS COMPLETE ---")
        print(f"Final graph: N={G.number_of_nodes()} E={G.number_of_edges()}")

    # 3) Build world, compute Λ⋆, quotient
    rule_factory = make_rule_factory(str(args.rule), threshold=int(args.threshold), alphabet_k=int(args.k), vacuum_fixed=int(args.vacuum))
    world = WorldInstance.homogeneous(G, k=int(args.k), rule_factory=rule_factory, seed=int(args.seed))

    scdc_cfg = SCDCConfig(
        seed=int(args.seed),
        max_iterations=int(args.scdc_iters),
        sample_tuples_per_vertex=int(args.scdc_samples),
        max_diamond_state_samples=int(args.scdc_diamond_samples),
    )
    profile = compute_lambda_star(world, cfg=scdc_cfg)
    loss = quotient_loss(world, profile)
    qworld = quotient_world(world, profile)

    # 4) Spacetime diagnostics
    growth = measure_light_cone_growth(G, t_max=int(args.tmax), n_sources=int(args.n_sources), seed=int(args.seed))
    cf = curvature_field_on_condensation(G, r_cone=int(args.r_cone))
    gap = _spectral_gap_sym(G)

    plot_spacetime_embedding(G, out_png=str(args.out_prefix) + "_spacetime.png")

    # 5) Matter injection + pocket tracking (optional)
    pocket_summary: Dict = {"found": False}
    matter_scc: Set[int] = set()

    if args.inject_knot_k > 0:
        nodes = list(G.nodes())
        knot_nodes = list(rng.choice(nodes, size=min(int(args.inject_knot_k), len(nodes)), replace=False))
        Gm = inject_knot(G, knot_nodes=knot_nodes, p_internal=float(args.knot_p), extra_parallel=int(args.knot_parallel), seed=int(args.seed))
        print(f"Injected knot: |K|={len(knot_nodes)}  E'={Gm.number_of_edges()}")

        world_m = WorldInstance.homogeneous(Gm, k=int(args.k), rule_factory=rule_factory, seed=int(args.seed))
        profile_m = compute_lambda_star(world_m, cfg=scdc_cfg)
        qworld_m = quotient_world(world_m, profile_m)

        # initial state: vacuum except knot nodes set to 1 (or random nonzero)
        x0 = world_m.vacuum_state(value=int(args.vacuum))
        for v in knot_nodes:
            x0[v] = 1 if int(args.k) == 2 else int(rng.integers(1, int(args.k)))
        x0q = quotient_state(x0, profile_m)

        ess = compute_essential_inputs(qworld_m, cfg=scdc_cfg)
        states = simulate(qworld_m, x0q, steps=int(args.steps), seed=int(args.seed), schedule_per_tick=True)

        pockets_over_time: List[Dict] = []
        best_pocket: Optional[Set[int]] = None
        best_size = 0

        for t, st in enumerate(states):
            active = active_set_from_state(st, vacuum_value=int(args.vacuum))
            pockets = pockets_from_active_set(qworld_m, active, ess)
            pockets = [p for p in pockets if p]  # drop empties
            pockets.sort(key=len, reverse=True)

            rec = {"t": int(t), "n_active": int(len(active)), "n_pockets": int(len(pockets)), "largest_pocket": int(len(pockets[0])) if pockets else 0}
            pockets_over_time.append(rec)

            if pockets and len(pockets[0]) > best_size:
                best_size = len(pockets[0])
                best_pocket = set(pockets[0])

        pocket_summary = {
            "found": bool(best_pocket is not None and best_size > 0),
            "best_size": int(best_size),
            "time_series": pockets_over_time,
            "knot_nodes": [int(v) for v in knot_nodes],
        }

        # map pocket nodes to condensation SCC ids
        if best_pocket is not None:
            cond = compute_condensation(Gm)
            matter_scc = {int(cond.node_to_scc[v]) for v in best_pocket}
            pocket_summary["matter_scc_size"] = int(len(matter_scc))
            pocket_summary["graph_with_matter_edges"] = int(Gm.number_of_edges())
            # overwrite G for geodesic plotting to include matter injection
            G = Gm

    # 6) Geodesics / lensing
    lens = plot_geodesic_lensing(G, matter_scc=matter_scc, out_png=str(args.out_prefix) + "_geodesics.png", n_pairs=int(args.n_pairs), seed=int(args.seed))

    # 7) Geodesic deviation overlap (two random nearby sources)
    cond = compute_condensation(G)
    dag = cond.dag
    nodes = list(dag.nodes())
    overlap = []
    if len(nodes) >= 2:
        s1 = int(nodes[0])
        # pick a neighbor if possible
        neighs = list(dag.successors(s1)) + list(dag.predecessors(s1))
        s2 = int(neighs[0]) if neighs else int(nodes[1])
        overlap = geodesic_deviation_overlap(dag, s1=s1, s2=s2, t_max=int(args.tmax))

    summary = {
        "graph_type": str(args.graph_type),
        "N": int(G.number_of_nodes()),
        "E": int(G.number_of_edges()),
        "rule": str(args.rule),
        "k": int(args.k),
        "scdc_loss": float(loss),
        "spectral_gap_sym": float(gap),
        "light_cones": [asdict(g) for g in growth],
        "curvature": {
            "mean_kappa_node": float(np.mean(list(cf.kappa_node.values()))) if cf.kappa_node else 0.0,
            "var_kappa_node": float(np.var(list(cf.kappa_node.values()))) if cf.kappa_node else 0.0,
            "mean_rho": float(np.mean(list(cf.rho.values()))) if cf.rho else 0.0,
            "var_rho": float(np.var(list(cf.rho.values()))) if cf.rho else 0.0,
        },
        "geodesic_overlap": overlap,
        "lensing": lens,
        "pockets": pocket_summary,
        "genesis": {
            "enabled": bool(args.genesis_steps > 0),
            "energy_mode": str(args.energy_mode),
            "history": [float(x) for x in genesis_hist[:200]] + (["..."] if len(genesis_hist) > 200 else []),
        },
        "artifacts": {
            "spacetime_png": str(args.out_prefix) + "_spacetime.png",
            "geodesics_png": str(args.out_prefix) + "_geodesics.png",
        },
    }
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified Consistency Universe: spacetime -> matter -> geodesics (SCDC-first).")

    # graph
    p.add_argument("--graph_type", choices=["er", "layered"], default="er")
    p.add_argument("--n", type=int, default=300, help="Number of nodes (or total nodes for layered).")
    p.add_argument("--p", type=float, default=0.02, help="ER edge probability.")
    p.add_argument("--max_parallel", type=int, default=1)

    # layered params
    p.add_argument("--layers", type=int, default=20)
    p.add_argument("--p_forward", type=float, default=0.08)
    p.add_argument("--p_skip2", type=float, default=0.01)
    p.add_argument("--p_skip3", type=float, default=0.0)

    # world rule
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--rule", choices=["threshold", "xor", "random"], default="threshold")
    p.add_argument("--threshold", type=int, default=1)
    p.add_argument("--vacuum", type=int, default=0)

    # scdc config
    p.add_argument("--scdc_iters", type=int, default=25)
    p.add_argument("--scdc_samples", type=int, default=1024)
    p.add_argument("--scdc_diamond_samples", type=int, default=64)

    # genesis
    p.add_argument("--genesis_steps", type=int, default=0, help="Set >0 to anneal the graph.")
    p.add_argument("--energy_mode", choices=["motif", "scdc"], default="motif")
    p.add_argument("--w_d", type=float, default=1.0)
    p.add_argument("--w_t", type=float, default=2.0)
    p.add_argument("--temp_start", type=float, default=2.0)
    p.add_argument("--cooling_rate", type=float, default=0.999)
    p.add_argument("--progress_every", type=int, default=250)

    # spacetime diagnostics
    p.add_argument("--tmax", type=int, default=8)
    p.add_argument("--n_sources", type=int, default=8)
    p.add_argument("--r_cone", type=int, default=1)

    # matter injection
    p.add_argument("--inject_knot_k", type=int, default=0)
    p.add_argument("--knot_p", type=float, default=0.9)
    p.add_argument("--knot_parallel", type=int, default=3)
    p.add_argument("--steps", type=int, default=80, help="Simulation steps for matter run.")

    # geodesics
    p.add_argument("--n_pairs", type=int, default=25)

    # io
    p.add_argument("--out_prefix", type=str, default="universe")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", type=str, default=None, help="Output JSON summary path (default: <out_prefix>_summary.json)")

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    # Default JSON output: keep each run's summary separate by prefix.
    if args.out_json is None:
        args.out_json = str(args.out_prefix) + "_summary.json"

    summary = run_pipeline(args)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {args.out_json}")
    print(f"Saved {args.out_prefix}_spacetime.png and {args.out_prefix}_geodesics.png")
    if summary.get("pockets", {}).get("found"):
        print(">> Matter pocket detected (see pockets time series in JSON).")
    else:
        print(">> No stable pocket detected with these parameters.")
    print(f"SCDC quotient loss: {summary['scdc_loss']:.4f}")
    if summary['light_cones']:
        models = [g['growth_model'] for g in summary['light_cones']]
        print(f"Light-cone growth models: {models}")
    print(f"Lensing mean min-distance: {summary['lensing'].get('mean_min_dist')}")
    print(f"Geodesic overlap (t=0..): {summary['geodesic_overlap']}")


if __name__ == "__main__":
    main()
