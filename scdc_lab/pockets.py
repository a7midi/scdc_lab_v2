from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import itertools
import networkx as nx
import numpy as np

from .world import WorldInstance
from .scdc import _sample_or_enumerate_tuples, _tuple_product_size, SCDCConfig


@dataclass
class EssentialInputs:
    """Stores essential-input information for all vertices."""
    ess_sources: Dict[int, Set[int]]  # v -> set of source vertices that are essential inputs


def compute_essential_inputs(world: WorldInstance, cfg: Optional[SCDCConfig] = None) -> EssentialInputs:
    """Compute EssIn(v) indirectly as the set of essential source vertices per v (Definition 4.1).

    We test essentiality by enumerating/sampling input tuples and checking whether varying a coordinate
    can change the output while holding other coordinates fixed.
    """
    if cfg is None:
        cfg = SCDCConfig()
    rng = np.random.default_rng(cfg.seed)

    ess_sources: Dict[int, Set[int]] = {v: set() for v in world.G.nodes()}

    for v in world.G.nodes():
        in_edges = world.in_edges[v]
        m = len(in_edges)
        if m == 0:
            continue

        # Enumerate/sampling over the full input space (could be big) to find a witness quickly.
        in_sizes = [world.alphabet_size[u] for (u, _key) in in_edges]
        tuples = _sample_or_enumerate_tuples(
            in_sizes,
            rng=rng,
            max_enumerate=cfg.max_enumerate_tuples_per_vertex,
            n_sample=cfg.sample_tuples_per_vertex,
        )

        # For each coordinate i, look for a pair that differs only at i and flips the output.
        # We do this by bucketing on all coordinates except i.
        for i, (src_u, _key) in enumerate(in_edges):
            # Map: other_coords -> (seen_value_at_i, output)
            seen: Dict[Tuple[int, ...], Dict[int, int]] = {}
            essential = False
            for t in tuples:
                key_other = tuple(t[j] for j in range(m) if j != i)
                val_i = int(t[i])
                out = int(world.local_rule[v](t))
                seen.setdefault(key_other, {})
                prev = seen[key_other].get(val_i)
                # store output for this i-value
                seen[key_other][val_i] = out
                # If among values of i (for fixed other coords), outputs disagree, essential.
                outs = set(seen[key_other].values())
                if len(outs) >= 2:
                    essential = True
                    break
            if essential:
                ess_sources[v].add(src_u)

    return EssentialInputs(ess_sources=ess_sources)


def predecessor_closure(world: WorldInstance, S: Set[int]) -> Set[int]:
    """PredCl(S) = union of predecessors iterated to closure."""
    pred: Dict[int, Set[int]] = {}
    for v in world.G.nodes():
        pred[v] = {u for (u, _key) in world.in_edges[v]}

    closure = set(S)
    changed = True
    while changed:
        changed = False
        to_add = set()
        for v in list(closure):
            to_add |= pred.get(v, set())
        if not to_add.issubset(closure):
            closure |= to_add
            changed = True
    return closure


def force_operator(world: WorldInstance, S: Set[int], ess: EssentialInputs) -> Set[int]:
    """Force(S) = {v: all essential input sources of v lie in PredCl(S)}."""
    predcl = predecessor_closure(world, S)
    forced = set()
    for v in world.G.nodes():
        srcs = ess.ess_sources.get(v, set())
        # IMPORTANT: ignore vertices with no essential inputs; otherwise empty EssIn forces everything
        # into every closure trivially.
        if srcs and srcs.issubset(predcl):
            forced.add(v)
    return forced


def closure(world: WorldInstance, S: Set[int], ess: EssentialInputs) -> Set[int]:
    """Cl(S) = least fixed point of C(S) = PredCl(S) ∪ Force(S) (Section 4.2)."""
    cur = set(S)
    while True:
        predcl = predecessor_closure(world, cur)
        forced = force_operator(world, cur, ess)
        nxt = predcl | forced
        if nxt == cur:
            return cur
        cur = nxt


def is_connected_undirected(world: WorldInstance, nodes: Set[int]) -> bool:
    if not nodes:
        return False
    UG = nx.Graph()
    UG.add_nodes_from(nodes)
    for u, v, _k in world.G.edges(keys=True):
        if u in nodes and v in nodes:
            UG.add_edge(u, v)
    return nx.is_connected(UG)


def is_autonomous_pocket(world: WorldInstance, P: Set[int], ess: EssentialInputs) -> bool:
    """Autonomous pocket iff Cl(P)=P and connected in underlying undirected graph (Definition 4.3)."""
    if not P:
        return False
    if closure(world, set(P), ess) != set(P):
        return False
    return is_connected_undirected(world, set(P))


def pockets_from_active_set(world: WorldInstance, active: Set[int], ess: EssentialInputs) -> List[Set[int]]:
    """Given an active node set, return closed connected pockets obtained by closure then splitting components."""
    if not active:
        return []
    P = closure(world, set(active), ess)
    # Split into connected components (undirected)
    UG = nx.Graph()
    UG.add_nodes_from(P)
    for u, v, _k in world.G.edges(keys=True):
        if u in P and v in P:
            UG.add_edge(u, v)
    comps = [set(c) for c in nx.connected_components(UG)]
    # Ensure each is closed (take closure component-wise)
    pockets = []
    for c in comps:
        Pc = closure(world, set(c), ess)
        pockets.append(Pc)
    return pockets
