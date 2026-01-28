from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import networkx as nx
import numpy as np

from .world import WorldInstance, make_rule_factory
from .scdc import SCDCConfig, compute_lambda_star, quotient_loss


# ----------------------------
# Motif-energy (proxy) Hamiltonian (legacy from the lab scripts)
# ----------------------------

@dataclass
class MotifCounts:
    diamonds: float
    ff_triangles: float


def motif_counts(G: nx.MultiDiGraph) -> MotifCounts:
    """Count diamonds and feed-forward triangles using sparse adjacency algebra.

    - Diamonds: converging length-2 paths: sum_{i,j} C((A^2)_{i,j}, 2)
    - FF triangles: feed-forward closure: sum_{i,j} (A^2 ⊙ A)_{i,j}

    Notes
    -----
    For MultiDiGraph, adjacency counts parallel edges. If you want a binarized
    adjacency, convert G to a simple DiGraph before calling this function.
    """
    import scipy.sparse as sp
    nodes = list(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=np.int64)
    A2 = A @ A

    data = A2.data
    mask = data >= 2
    k = data[mask].astype(float)
    diamonds = float(np.sum(k * (k - 1.0) / 2.0))

    overlap = A2.multiply(A)
    ff_triangles = float(overlap.sum())
    return MotifCounts(diamonds=diamonds, ff_triangles=ff_triangles)


def motif_energy(G: nx.MultiDiGraph, w_d: float = 1.0, w_t: float = 2.0) -> float:
    c = motif_counts(G)
    return - (float(w_d) * c.diamonds + float(w_t) * c.ff_triangles)


# ----------------------------
# Pure-consistency energy via Λ⋆ quotient loss
# ----------------------------

@dataclass
class ConsistencyEnergyConfig:
    alphabet_k: int = 2
    rule: str = "threshold"
    threshold: int = 1
    vacuum_fixed: int = 0

    # Use default_factory to avoid mutable default issues
    scdc: SCDCConfig = field(default_factory=lambda: SCDCConfig(
        max_iterations=25,
        sample_tuples_per_vertex=1024,
        max_diamond_state_samples=64,
        seed=0,
    ))


def consistency_energy(G: nx.MultiDiGraph, cfg: ConsistencyEnergyConfig) -> float:
    """Energy = quotient loss after enforcing consistency Λ⋆."""
    rule_factory = make_rule_factory(
        cfg.rule,
        threshold=int(cfg.threshold),
        alphabet_k=int(cfg.alphabet_k),
        vacuum_fixed=int(cfg.vacuum_fixed),
    )
    world = WorldInstance.homogeneous(G, k=int(cfg.alphabet_k), rule_factory=rule_factory, seed=cfg.scdc.seed)
    profile = compute_lambda_star(world, cfg=cfg.scdc)
    return quotient_loss(world, profile)


# ----------------------------
# Annealing / Genesis
# ----------------------------

@dataclass
class AnnealConfig:
    steps: int = 5000
    temp_start: float = 2.0
    cooling_rate: float = 0.999
    seed: int = 0
    forbid_self_loops: bool = True


def propose_rewire_aggressive(G: nx.MultiDiGraph, rng: np.random.Generator, forbid_self_loops: bool = True) -> nx.MultiDiGraph:
    """Rewire by removing one random edge and adding one random edge with the same source."""
    if G.number_of_edges() == 0:
        return G
    H = G.copy()
    nodes = list(H.nodes())
    edges = list(H.edges(keys=True))
    u, v, k = edges[int(rng.integers(0, len(edges)))]
    H.remove_edge(u, v, key=k)

    for _ in range(50):
        new_v = nodes[int(rng.integers(0, len(nodes)))]
        if forbid_self_loops and int(u) == int(new_v):
            continue
        H.add_edge(u, int(new_v))
        break
    return H


def run_genesis(
    G0: nx.MultiDiGraph,
    energy_fn: Callable[[nx.MultiDiGraph], float],
    cfg: AnnealConfig,
    progress_every: int = 250,
) -> Tuple[nx.MultiDiGraph, List[float]]:
    rng = np.random.default_rng(int(cfg.seed))
    G = G0.copy()
    E = float(energy_fn(G))
    temp = float(cfg.temp_start)
    hist: List[float] = [E]

    for t in range(int(cfg.steps)):
        H = propose_rewire_aggressive(G, rng, forbid_self_loops=cfg.forbid_self_loops)
        E2 = float(energy_fn(H))
        delta = E2 - E

        if delta < 0 or rng.random() < float(np.exp(-delta / max(1e-12, temp))):
            G, E = H, E2

        temp *= float(cfg.cooling_rate)
        hist.append(E)

        if progress_every and (t % int(progress_every) == 0):
            print(f"Step {t:6d}: T={temp:.4f}  E={E:.6f}")

    return G, hist
