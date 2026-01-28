from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np


def er_directed_multigraph(
    n: int,
    p: float,
    seed: Optional[int] = None,
    allow_self_loops: bool = False,
    max_parallel: int = 1,
) -> nx.MultiDiGraph:
    """Directed Erdős–Rényi MultiDiGraph.

    For each ordered pair (u,v) we add k parallel edges where k is:
      - 0 with prob (1-p)
      - 1..max_parallel with prob p/max_parallel each (uniform over 1..max_parallel)

    This yields controllable multi-edge multiplicity (useful for star-equivariance).
    """
    # Backwards-compatibility: allow a single p_skip to mean a 2-layer skip probability.
    if p_skip is not None:
        p_skip2 = float(p_skip)
    rng = np.random.default_rng(seed)
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n))
    for u in range(n):
        for v in range(n):
            if (not allow_self_loops) and (u == v):
                continue
            if rng.random() < p:
                k = int(rng.integers(1, max_parallel + 1))
                for _ in range(k):
                    G.add_edge(u, v)
    return G


def layered_random_dag(
    n_layers: int,
    layer_size: int,
    p_forward: float,
    p_skip2: float = 0.0,
    p_skip3: float = 0.0,
    p_skip: Optional[float] = None,  # alias for p_skip2
    seed: Optional[int] = None,
) -> nx.MultiDiGraph:
    """Layered DAG for causal / glider-like experiments.

    Nodes are labeled by integer id = layer*layer_size + i.

    Edges:
    - forward:  ℓ -> ℓ+1 with prob p_forward
    - skip2:    ℓ -> ℓ+2 with prob p_skip2
    - skip3:    ℓ -> ℓ+3 with prob p_skip3
    """
    rng = np.random.default_rng(seed)
    n = n_layers * layer_size
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n))

    def nid(layer: int, i: int) -> int:
        return layer * layer_size + i

    for layer in range(n_layers):
        for i in range(layer_size):
            u = nid(layer, i)
            # forward
            if layer + 1 < n_layers:
                for j in range(layer_size):
                    if rng.random() < p_forward:
                        G.add_edge(u, nid(layer + 1, j))
            # skip2
            if p_skip2 > 0 and layer + 2 < n_layers:
                for j in range(layer_size):
                    if rng.random() < p_skip2:
                        G.add_edge(u, nid(layer + 2, j))
            # skip3
            if p_skip3 > 0 and layer + 3 < n_layers:
                for j in range(layer_size):
                    if rng.random() < p_skip3:
                        G.add_edge(u, nid(layer + 3, j))
    return G


def inject_knot(
    G: nx.MultiDiGraph,
    knot_nodes: Sequence[int],
    p_internal: float = 0.8,
    extra_parallel: int = 2,
    add_back_edges: bool = True,
    seed: Optional[int] = None,
) -> nx.MultiDiGraph:
    """Inject a dense 'knot' into G by adding many internal edges.

    This is intentionally agnostic about 'physics'; it is just a way to create a localized
    topological defect that SCDC may or may not quarantine into a pocket.
    """
    rng = np.random.default_rng(seed)
    H = G.copy()
    K = list(knot_nodes)
    for u in K:
        for v in K:
            if u == v:
                continue
            if rng.random() < p_internal:
                k = int(rng.integers(1, extra_parallel + 1))
                for _ in range(k):
                    H.add_edge(u, v)
            if add_back_edges and rng.random() < (p_internal * 0.25):
                k = int(rng.integers(1, max(2, extra_parallel + 1)))
                for _ in range(k):
                    H.add_edge(v, u)
    return H


@dataclass(frozen=True)
class Condensation:
    """Condensation information for a directed graph."""

    sccs: List[Set[int]]
    node_to_scc: Dict[int, int]
    dag: nx.DiGraph
    depth: Dict[int, int]


def compute_condensation(G: nx.MultiDiGraph) -> Condensation:
    """Compute SCCs, condensation DAG, and a longest-path depth label on the DAG."""
    # networkx condensation works on DiGraph; for MultiDiGraph, convert to DiGraph (parallel edges ignored)
    simple = nx.DiGraph()
    simple.add_nodes_from(G.nodes())
    simple.add_edges_from((u, v) for (u, v, _k) in G.edges(keys=True))

    scc_list = [set(c) for c in nx.strongly_connected_components(simple)]
    node_to_scc: Dict[int, int] = {}
    for idx, comp in enumerate(scc_list):
        for node in comp:
            node_to_scc[node] = idx

    dag = nx.DiGraph()
    dag.add_nodes_from(range(len(scc_list)))
    for u, v in simple.edges():
        su, sv = node_to_scc[u], node_to_scc[v]
        if su != sv:
            dag.add_edge(su, sv)

    if not nx.is_directed_acyclic_graph(dag):
        # condensation should be a DAG by construction; if it isn't, something is wrong
        raise RuntimeError("Condensation DAG is not acyclic (unexpected).")

    depth = longest_path_depth(dag)
    return Condensation(sccs=scc_list, node_to_scc=node_to_scc, dag=dag, depth=depth)


def longest_path_depth(dag: nx.DiGraph) -> Dict[int, int]:
    """Depth label = length of a longest directed path ending at node."""
    order = list(nx.topological_sort(dag))
    depth: Dict[int, int] = {v: 0 for v in dag.nodes()}
    for v in order:
        preds = list(dag.predecessors(v))
        depth[v] = 0 if not preds else 1 + max(depth[p] for p in preds)
    return depth
