from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import math
import networkx as nx
import numpy as np

from .graphs import compute_condensation
from .diagnostics import cone_volumes, edge_curvature, memory_density


@dataclass
class LightConeGrowth:
    source: int
    volumes: List[int]          # V(t) = |I^+_t(source)|
    t_values: List[int]
    growth_model: str           # 'polynomial' or 'exponential' or 'unclear'
    r2_poly: float
    r2_exp: float
    eff_dim: Optional[float]


def light_cone_volumes(dag: nx.DiGraph, source: int, t_max: int) -> List[int]:
    """Compute V(t) = |I^+_t(source)| for t=0..t_max in a DAG."""
    source = int(source)
    seen = {source}
    frontier = {source}
    vols = [1]
    for _t in range(int(t_max)):
        nxt = set()
        for u in frontier:
            for v in dag.successors(u):
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        frontier = nxt
        vols.append(int(len(seen)))
        if not frontier:
            # no further growth
            # pad remaining with constant
            for _ in range(_t + 1, int(t_max)):
                vols.append(int(len(seen)))
            break
    return vols[: int(t_max) + 1]


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.array(y, dtype=float)
    yhat = np.array(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / max(1e-12, ss_tot)


def _fit_poly(vols: Sequence[int], tmin: int = 1) -> Tuple[float, float]:
    """Fit log V ~ d log t (power law). Returns (d, r2)."""
    t = np.arange(len(vols), dtype=float)
    V = np.array(vols, dtype=float)
    mask = (t >= float(tmin)) & (V > 0)
    if int(np.sum(mask)) < 3:
        return float("nan"), float("-inf")
    x = np.log(t[mask])
    y = np.log(V[mask])
    d, b = np.polyfit(x, y, 1)
    yhat = d * x + b
    return float(d), float(_r2(y, yhat))


def _fit_exp(vols: Sequence[int], tmin: int = 1) -> Tuple[float, float]:
    """Fit log V ~ α t (exponential). Returns (α, r2)."""
    t = np.arange(len(vols), dtype=float)
    V = np.array(vols, dtype=float)
    mask = (t >= float(tmin)) & (V > 0)
    if int(np.sum(mask)) < 3:
        return float("nan"), float("-inf")
    x = t[mask]
    y = np.log(V[mask])
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    return float(a), float(_r2(y, yhat))


def characterize_growth(vols: Sequence[int]) -> Tuple[str, float, float, Optional[float]]:
    """Classify growth as polynomial vs exponential (heuristic)."""
    d, r2_poly = _fit_poly(vols, tmin=2)
    a, r2_exp = _fit_exp(vols, tmin=2)

    # pick best model if it is significantly better
    if r2_poly > r2_exp + 0.05:
        return "polynomial", float(r2_poly), float(r2_exp), (None if not math.isfinite(d) else float(d))
    if r2_exp > r2_poly + 0.05:
        return "exponential", float(r2_poly), float(r2_exp), None

    # if both good, default to polynomial (more "geometric")
    if r2_poly > 0.85 and r2_exp > 0.85:
        return "polynomial", float(r2_poly), float(r2_exp), (None if not math.isfinite(d) else float(d))

    return "unclear", float(r2_poly), float(r2_exp), (None if not math.isfinite(d) else float(d))


def measure_light_cone_growth(
    G: nx.MultiDiGraph,
    t_max: int = 8,
    n_sources: int = 8,
    seed: int = 0,
) -> List[LightConeGrowth]:
    """Measure light cone growth on the condensation DAG for multiple sources."""
    rng = np.random.default_rng(int(seed))
    cond = compute_condensation(G)
    dag = cond.dag
    nodes = list(dag.nodes())
    if not nodes:
        return []
    sources = list(rng.choice(nodes, size=min(int(n_sources), len(nodes)), replace=False))
    out: List[LightConeGrowth] = []
    for s in sources:
        vols = light_cone_volumes(dag, int(s), int(t_max))
        model, r2_poly, r2_exp, eff_dim = characterize_growth(vols)
        out.append(LightConeGrowth(
            source=int(s),
            volumes=[int(v) for v in vols],
            t_values=list(range(len(vols))),
            growth_model=model,
            r2_poly=float(r2_poly),
            r2_exp=float(r2_exp),
            eff_dim=(None if eff_dim is None else float(eff_dim)),
        ))
    return out


@dataclass
class CurvatureField:
    V: Dict[int, int]
    kappa_edge: Dict[Tuple[int, int], int]
    rho: Dict[int, int]
    kappa_node: Dict[int, float]


def curvature_field_on_condensation(G: nx.MultiDiGraph, r_cone: int = 1) -> CurvatureField:
    """Compute cone-volume curvature κ_r and memory density ρ on condensation DAG."""
    cond = compute_condensation(G)
    dag = cond.dag
    V = cone_volumes(dag, r=int(r_cone))
    kappa_edge = edge_curvature(dag, V)
    rho = memory_density(dag)
    kappa_node: Dict[int, float] = {}
    for u in dag.nodes():
        outs = [kappa_edge[(u, v)] for v in dag.successors(u)]
        kappa_node[int(u)] = float(np.mean(outs)) if outs else 0.0
    return CurvatureField(V=V, kappa_edge=kappa_edge, rho=rho, kappa_node=kappa_node)


# ----------------------------
# Embedding + geodesics
# ----------------------------

def _largest_undirected_component_nodes(dag: nx.DiGraph) -> List[int]:
    UG = dag.to_undirected()
    comps = list(nx.connected_components(UG))
    if not comps:
        return []
    comps.sort(key=len, reverse=True)
    return sorted(int(x) for x in comps[0])


def all_pairs_shortest_path_lengths_undirected(dag: nx.DiGraph, nodes: Optional[Sequence[int]] = None) -> Tuple[List[int], np.ndarray]:
    """All-pairs shortest-path lengths on the undirected projection of a DAG.

    Returns (node_list, dist_matrix). dist_matrix[i,j] = distance, or large value if disconnected.
    """
    UG = dag.to_undirected()
    if nodes is None:
        nodes = _largest_undirected_component_nodes(dag)
    nodes = list(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    dist = np.full((n, n), fill_value=float(n + 1), dtype=float)
    for i in range(n):
        dist[i, i] = 0.0
    for s in nodes:
        i = idx[s]
        lengths = nx.single_source_shortest_path_length(UG, s)
        for t, d in lengths.items():
            if t in idx:
                dist[i, idx[t]] = float(d)
    return nodes, dist


def classical_mds_2d(dist: np.ndarray) -> np.ndarray:
    """Classical MDS into 2D using double-centering of squared distances."""
    D2 = dist ** 2
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / float(n)
    B = -0.5 * J @ D2 @ J
    # eigen-decomposition
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    # take top 2 nonnegative components
    w = np.maximum(evals[:2], 0.0)
    X = evecs[:, :2] * np.sqrt(w.reshape(1, -1))
    return X


def embed_condensation_mds(G: nx.MultiDiGraph) -> Tuple[List[int], np.ndarray]:
    cond = compute_condensation(G)
    nodes, dist = all_pairs_shortest_path_lengths_undirected(cond.dag)
    X = classical_mds_2d(dist)
    return nodes, X


def shortest_path_geodesic(
    dag: nx.DiGraph,
    s: int,
    t: int,
    undirected: bool = True,
) -> Optional[List[int]]:
    """Geodesic = shortest path (hop metric) in directed or undirected projection."""
    s = int(s)
    t = int(t)
    H = dag.to_undirected() if undirected else dag
    try:
        return [int(x) for x in nx.shortest_path(H, s, t)]
    except Exception:
        return None


def lensing_score_for_path(path: Sequence[int], matter: Set[int], dag: nx.DiGraph) -> Tuple[int, float]:
    """Return (min_graph_distance_to_matter, mean_graph_distance_to_matter) along a path."""
    if not path:
        return 10**9, float("inf")
    UG = dag.to_undirected()
    # precompute distances from matter set (multi-source BFS)
    dist = {m: 0 for m in matter}
    q = list(matter)
    # BFS
    for m in q:
        for nb in UG.neighbors(m):
            if nb not in dist:
                dist[nb] = 1
                q.append(nb)
    dvals = [dist.get(int(v), 10**9) for v in path]
    return int(min(dvals)), float(np.mean(dvals))


def geodesic_deviation_overlap(
    dag: nx.DiGraph,
    s1: int,
    s2: int,
    t_max: int = 10,
) -> List[float]:
    """A simple geodesic-deviation proxy via overlap of future cones.

    For each t:
      overlap(t) = |I^+_t(s1) ∩ I^+_t(s2)| / min(|I^+_t(s1)|, |I^+_t(s2)|)

    In flat geometries, overlap tends to remain high for nearby sources; curvature/defects
    can reduce overlap (divergence) or increase it (convergence).
    """
    s1 = int(s1)
    s2 = int(s2)

    def cone(s: int, t: int) -> Set[int]:
        seen = {s}
        frontier = {s}
        for _ in range(int(t)):
            nxt = set()
            for u in frontier:
                for v in dag.successors(u):
                    if v not in seen:
                        seen.add(v)
                        nxt.add(v)
            frontier = nxt
            if not frontier:
                break
        return seen

    overlaps: List[float] = []
    for t in range(int(t_max) + 1):
        c1 = cone(s1, t)
        c2 = cone(s2, t)
        denom = float(min(len(c1), len(c2)))
        overlaps.append(float(len(c1 & c2) / max(1.0, denom)))
    return overlaps
