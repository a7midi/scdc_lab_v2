from __future__ import annotations

"""Hypothesis 3: Generational bands via spectral clustering.

We compare the condensation spectrum of:
- a base graph (possibly genesis-produced)
- degree-preserving nulls
- depth-preserving nulls (using frozen condensation-depth labels)

Outputs a JSON report with best-k and cluster centers for singular values and symmetric eigenvalue magnitudes.
"""

import argparse
import json
from typing import Any, Dict, List

import numpy as np

def _unpack_swap_result(res, fallback_graph):
    """Handle multiple implementations of *double_edge_swap utilities.

    Some versions return:
      - H
      - (H, swaps_done)
      - (H, swaps_done, tries, ...)
      - None (in-place modification)
    """
    if res is None:
        return fallback_graph, None
    if isinstance(res, (tuple, list)):
        if len(res) == 0:
            return fallback_graph, None
        H = res[0]
        swaps_done = None
        if len(res) > 1 and isinstance(res[1], (int, np.integer)):
            swaps_done = int(res[1])
        return H, swaps_done
    return res, None


from ..graphs import er_directed_multigraph, layered_random_dag
from ..diagnostics import directed_double_edge_swap, frozen_depth_labels_from_condensation, depth_preserving_double_edge_swap
from ..spectral import generation_band_test


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="H3: Generations via spectral bands.")
    p.add_argument("--graph_type", choices=["er", "layered"], default="er")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--p", type=float, default=0.02)
    p.add_argument("--layers", type=int, default=20)
    p.add_argument("--p_forward", type=float, default=0.08)
    p.add_argument("--p_skip2", type=float, default=0.01)
    p.add_argument("--p_skip3", type=float, default=0.0)

    p.add_argument("--n_null", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--swap_factor", type=float, default=1.0, help="Number of swaps = swap_factor * E.")
    return p


def summarize_test(tag: str, test) -> Dict[str, Any]:
    out: Dict[str, Any] = {"tag": tag}
    if test.clustering_svals is not None:
        out["svals_best_k"] = int(test.clustering_svals.k)
        out["svals_score"] = float(test.clustering_svals.score)
        out["svals_centers"] = [float(x) for x in np.sort(test.clustering_svals.centers)]
    if test.clustering_evals_sym is not None:
        out["sym_best_k"] = int(test.clustering_evals_sym.k)
        out["sym_score"] = float(test.clustering_evals_sym.score)
        out["sym_centers"] = [float(x) for x in np.sort(test.clustering_evals_sym.centers)]
    return out


def main() -> None:
    args = build_argparser().parse_args()
    rng = np.random.default_rng(int(args.seed))

    if args.graph_type == "er":
        G = er_directed_multigraph(n=int(args.n), p=float(args.p), seed=int(args.seed), max_parallel=1)
    else:
        layer_size = int(args.n // max(1, int(args.layers)))
        G = layered_random_dag(n_layers=int(args.layers), layer_size=layer_size, p_forward=float(args.p_forward), p_skip2=float(args.p_skip2), p_skip3=float(args.p_skip3), seed=int(args.seed))

    E = G.number_of_edges()
    n_swaps = int(max(1, float(args.swap_factor) * E))

    report: Dict[str, Any] = {}
    report["base"] = summarize_test("base", generation_band_test(G, random_state=int(args.seed)))

    # Degree nulls
    report["deg_nulls"] = []
    for i in range(int(args.n_null)):
        G0 = G.copy()
        res = directed_double_edge_swap(G0, n_swaps=n_swaps, seed=int(args.seed) + 1000 + i)
        H, swaps_done = _unpack_swap_result(res, G0)
        report["deg_nulls"].append(summarize_test(f"deg_null_{i}", generation_band_test(H, random_state=int(args.seed) + 1000 + i)))

    # Depth nulls
    frozen_depth = frozen_depth_labels_from_condensation(G)
    report["depth_nulls"] = []
    for i in range(int(args.n_null)):
        G0 = G.copy()
        res = depth_preserving_double_edge_swap(G0, frozen_depth=frozen_depth, n_swaps=n_swaps, seed=int(args.seed) + 2000 + i)
        H, swaps_done = _unpack_swap_result(res, G0)
        report["depth_nulls"].append(summarize_test(f"depth_null_{i}", generation_band_test(H, random_state=int(args.seed) + 2000 + i)))

    report["n_swaps"] = int(n_swaps)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()