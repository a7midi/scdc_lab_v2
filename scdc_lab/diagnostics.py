from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math
import networkx as nx
import numpy as np
from scipy import stats

from .graphs import compute_condensation


@dataclass
class CurvMemBlock:
    nodes: List[int]              # SCC node ids in condensation DAG
    depth: int
    rho_R: float
    kappa_R: float
    n_internal_edges: int


@dataclass
class CurvMemFit:
    R: int
    n_blocks: int
    a_ls: float
    b_ls: float
    a_rob: float
    b_rob: float


@dataclass
class PlateauResult:
    ok: bool
    window: Optional[Tuple[int, int]]  # (R_min, R_max)
    a_star_rob: Optional[float]
    a_star_ls: Optional[float]
    delta_a_star: Optional[float]
    fits: List[CurvMemFit]


def cone_volumes(dag: nx.DiGraph, r: int = 1) -> Dict[int, int]:
    """Compute Vr(X) = |I^+_r(X)| on a DAG."""
    V = {}
    for x in dag.nodes():
        # BFS to depth r
        seen = {x}
        frontier = {x}
        for _ in range(r):
            nxt = set()
            for u in frontier:
                for v in dag.successors(u):
                    if v not in seen:
                        seen.add(v)
                        nxt.add(v)
            frontier = nxt
        V[x] = len(seen)
    return V


def edge_curvature(dag: nx.DiGraph, V: Dict[int, int]) -> Dict[Tuple[int, int], int]:
    """κ_r(X->Y) = V(Y) - V(X)."""
    kappa = {}
    for u, v in dag.edges():
        kappa[(u, v)] = int(V[v] - V[u])
    return kappa


def memory_density(dag: nx.DiGraph) -> Dict[int, int]:
    """ρ_mem(X) = outdeg(X) - 1."""
    return {x: int(dag.out_degree(x) - 1) for x in dag.nodes()}


def depth_slices(depth: Dict[int, int]) -> Dict[int, List[int]]:
    slices: Dict[int, List[int]] = {}
    for node, d in depth.items():
        slices.setdefault(d, []).append(node)
    for d in slices:
        slices[d].sort()
    return slices


def blocks_by_depth(dag: nx.DiGraph, depth: Dict[int, int], R: int) -> List[List[int]]:
    """Deterministically partition each depth slice into contiguous blocks of size ~R."""
    blocks: List[List[int]] = []
    slices = depth_slices(depth)
    for d, nodes in sorted(slices.items()):
        for i in range(0, len(nodes), R):
            blocks.append(nodes[i:i + R])
    return blocks


def block_observables(
    dag: nx.DiGraph,
    depth: Dict[int, int],
    R: int,
    r_cone: int = 1,
) -> List[CurvMemBlock]:
    V = cone_volumes(dag, r=r_cone)
    kappa_e = edge_curvature(dag, V)
    rho = memory_density(dag)

    out: List[CurvMemBlock] = []
    blocks = blocks_by_depth(dag, depth, R)

    for block in blocks:
        if not block:
            continue
        d = depth[block[0]]
        # node-average rho
        rho_R = float(np.mean([rho[x] for x in block]))
        # edge-average kappa over internal edges
        internal_edges = [(u, v) for (u, v) in dag.edges() if u in block and v in block]
        if not internal_edges:
            continue
        kappa_R = float(np.mean([kappa_e[(u, v)] for (u, v) in internal_edges]))
        out.append(CurvMemBlock(nodes=block, depth=d, rho_R=rho_R, kappa_R=kappa_R, n_internal_edges=len(internal_edges)))
    return out


def fit_affine(blocks: List[CurvMemBlock]) -> Optional[Tuple[float, float, float, float]]:
    """Fit κ ≈ a ρ + b by LS and Theil–Sen. Returns (a_ls,b_ls,a_rob,b_rob) or None."""
    xs = np.array([b.rho_R for b in blocks], dtype=float)
    ys = np.array([b.kappa_R for b in blocks], dtype=float)
    if len(xs) < 3:
        return None
    # Require some variation in x to avoid degeneracy
    if float(np.std(xs)) < 1e-12:
        return None
    # LS
    a_ls, b_ls = np.polyfit(xs, ys, 1)
    # Theil–Sen (robust)
    ts = stats.theilslopes(ys, xs)
    a_rob = float(ts[0])
    b_rob = float(ts[1])
    return float(a_ls), float(b_ls), float(a_rob), float(b_rob)


def compute_aR_curve(
    G: nx.MultiDiGraph,
    R_values: Sequence[int],
    min_blocks: int = 12,
    r_cone: int = 1,
) -> List[CurvMemFit]:
    cond = compute_condensation(G)
    dag = cond.dag
    depth = cond.depth
    fits: List[CurvMemFit] = []
    for R in R_values:
        blocks = block_observables(dag, depth, R, r_cone=r_cone)
        # Domain: rho_R > 0 and enough blocks
        blocks = [b for b in blocks if b.rho_R > 0]
        if len(blocks) < min_blocks:
            continue
        fit = fit_affine(blocks)
        if fit is None:
            continue
        a_ls, b_ls, a_rob, b_rob = fit
        fits.append(CurvMemFit(R=int(R), n_blocks=len(blocks), a_ls=a_ls, b_ls=b_ls, a_rob=a_rob, b_rob=b_rob))
    return fits


def find_plateau(
    fits: List[CurvMemFit],
    window_len: int = 3,
    rel_tol: float = 0.07,
) -> PlateauResult:
    """Find the first plateau window scanning from largest R down.

    A window [R_i,...,R_{i+L-1}] is accepted if:
      - we have at least window_len points
      - robust slopes a_rob are relatively stable: (max-min)/max(1e-9,|median|) < rel_tol
    """
    if not fits:
        return PlateauResult(ok=False, window=None, a_star_rob=None, a_star_ls=None, delta_a_star=None, fits=[])

    fits_sorted = sorted(fits, key=lambda f: f.R)
    # scan windows ending at the largest R
    for end in range(len(fits_sorted) - 1, window_len - 2, -1):
        win = fits_sorted[end - window_len + 1 : end + 1]
        a_vals = np.array([w.a_rob for w in win], dtype=float)
        med = float(np.median(a_vals))
        denom = max(1e-9, abs(med))
        rel = float((np.max(a_vals) - np.min(a_vals)) / denom)
        if rel <= rel_tol:
            a_star_rob = float(np.median([w.a_rob for w in win]))
            a_star_ls = float(np.median([w.a_ls for w in win]))
            delta = float(a_star_ls - a_star_rob)
            return PlateauResult(
                ok=True,
                window=(win[0].R, win[-1].R),
                a_star_rob=a_star_rob,
                a_star_ls=a_star_ls,
                delta_a_star=delta,
                fits=fits_sorted,
            )

    return PlateauResult(ok=False, window=None, a_star_rob=None, a_star_ls=None, delta_a_star=None, fits=fits_sorted)


# ----------------------------
# Null models (rewiring)
# ----------------------------

def _edge_list(G: nx.MultiDiGraph) -> List[Tuple[int, int, int]]:
    return [(u, v, k) for (u, v, k) in G.edges(keys=True)]


def directed_double_edge_swap(
    G: nx.MultiDiGraph,
    n_swaps: int,
    seed: Optional[int] = None,
    forbid_self_loops: bool = True,
) -> nx.MultiDiGraph:
    """Degree-preserving directed double-edge swaps (Null-1 style).

    Pick two edges (a->b) and (c->d) and swap targets to (a->d) and (c->b).
    In/out degrees of all nodes are preserved.
    """
    rng = np.random.default_rng(seed)
    H = G.copy()
    for _ in range(n_swaps):
        edges = _edge_list(H)
        if len(edges) < 2:
            break
        i, j = rng.choice(len(edges), size=2, replace=False)
        a, b, k1 = edges[i]
        c, d, k2 = edges[j]
        # Proposed swapped edges
        if forbid_self_loops and (a == d or c == b):
            continue
        # Remove original edges
        try:
            H.remove_edge(a, b, key=k1)
            H.remove_edge(c, d, key=k2)
        except Exception:
            continue
        # Add swapped
        H.add_edge(a, d)
        H.add_edge(c, b)
    return H


def depth_preserving_double_edge_swap(
    G: nx.MultiDiGraph,
    frozen_depth: Dict[int, int],
    n_swaps: int,
    seed: Optional[int] = None,
    forbid_self_loops: bool = True,
    max_tries_factor: int = 50,
) -> nx.MultiDiGraph:
    """Depth-label preserving swaps (Null-2 style from the Letter).

    Only swap edges within the same (depth(source), depth(target)) bucket.

    IMPORTANT FIX:
      The old implementation iterated exactly `n_swaps` times but *did not guarantee* it actually
      performed `n_swaps` successful swaps (it would `continue` on failed proposals).
      This makes the null model under-randomized, especially for layered graphs.

    This version attempts up to `max_tries_factor * n_swaps` proposals and stops once
    `n_swaps` swaps have been ACCEPTED (or no viable buckets exist).
    """
    rng = np.random.default_rng(seed)
    H = G.copy()

    # Build buckets of edges by depth-pair once; maintain as lists of (u, v, key).
    buckets: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    for u, v, k in H.edges(keys=True):
        du = frozen_depth.get(u, 0)
        dv = frozen_depth.get(v, 0)
        buckets.setdefault((du, dv), []).append((u, v, k))

    # Keep a cached list of bucket keys that currently have >=2 edges.
    def refresh_candidates() -> List[Tuple[int, int]]:
        return [pair for pair, eds in buckets.items() if len(eds) >= 2]

    accepted = 0
    max_tries = max(1, int(max_tries_factor * max(1, n_swaps)))
    tries = 0

    while accepted < n_swaps and tries < max_tries:
        tries += 1
        candidates = refresh_candidates()
        if not candidates:
            break

        pair = candidates[int(rng.integers(0, len(candidates)))]
        eds = buckets[pair]
        # Choose two distinct edges within the same depth bucket.
        i, j = rng.choice(len(eds), size=2, replace=False)
        a, b, k1 = eds[i]
        c, d, k2 = eds[j]

        # Proposed swapped edges: a->d and c->b
        if forbid_self_loops and (a == d or c == b):
            continue

        # Remove the two chosen edges (by key) if they still exist.
        try:
            H.remove_edge(a, b, key=k1)
            H.remove_edge(c, d, key=k2)
        except Exception:
            # Buckets might be stale if keys got invalid; rebuild once and continue.
            buckets.clear()
            for u, v, k in H.edges(keys=True):
                du = frozen_depth.get(u, 0)
                dv = frozen_depth.get(v, 0)
                buckets.setdefault((du, dv), []).append((u, v, k))
            continue

        # Add swapped edges (new keys auto-assigned by MultiDiGraph)
        k1_new = H.add_edge(a, d)
        k2_new = H.add_edge(c, b)

        # Update bucket lists in-place.
        # Remove entries at indices i and j (remove larger index first).
        for idx in sorted([i, j], reverse=True):
            eds.pop(idx)
        eds.append((a, d, k1_new))
        eds.append((c, b, k2_new))

        accepted += 1

    return H


def frozen_depth_labels_from_condensation(G: nx.MultiDiGraph) -> Dict[int, int]:
    """Assign each vertex the depth of its SCC in the condensation DAG (frozen labels)."""
    cond = compute_condensation(G)
    return {v: int(cond.depth[cond.node_to_scc[v]]) for v in G.nodes()}
