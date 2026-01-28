from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import itertools
import numpy as np
import networkx as nx

from .partitions import Partition, profile_equal
from .graphs import compute_condensation
from .world import WorldInstance, LocalRule, LookupTableRule


class LazyQuotientRule(LocalRule):
    r"""On-demand quotient rule \bar\lambda_v.

    We avoid enumerating the full quotient input space Π_e |\bar A_{s(e)}|, which can explode.
    """

    def __init__(
        self,
        *,
        in_edges: Sequence[Tuple[int, int]],
        reps_per_coord: Sequence[Dict[int, int]],
        q_out_map: Dict[int, int],
        raw_rule: LocalRule,
        cache_limit: int = 200_000,
    ):
        self._in_edges = list(in_edges)
        self._reps = list(reps_per_coord)
        self._q_out = dict(q_out_map)
        self._raw_rule = raw_rule
        self._cache_limit = int(cache_limit)
        self._cache: Dict[Tuple[int, ...], int] = {}

    def __call__(self, inputs: Tuple[int, ...]) -> int:
        key = tuple(int(x) for x in inputs)
        if key in self._cache:
            return int(self._cache[key])

        raw_inputs = tuple(self._reps[i][key[i]] for i in range(len(self._in_edges)))
        raw_out = int(self._raw_rule(raw_inputs))
        out = int(self._q_out[raw_out])

        if len(self._cache) < self._cache_limit:
            self._cache[key] = out
        return out


@dataclass
class SCDCConfig:
    # tuple enumeration / sampling for admissibility + essential inputs
    max_enumerate_tuples_per_vertex: int = 4096
    sample_tuples_per_vertex: int = 4096

    # diamond sampling for confluence
    max_diamond_state_samples: int = 256

    # fixed point iteration
    max_iterations: int = 50

    # random seed
    seed: Optional[int] = None


def discrete_profile(world: WorldInstance) -> Dict[int, Partition]:
    return {v: Partition.discrete(world.alphabet_size[v]) for v in world.G.nodes()}


def _tuple_product_size(sizes: Sequence[int]) -> int:
    out = 1
    for s in sizes:
        out *= int(s)
    return out


def _sample_or_enumerate_tuples(
    sizes: Sequence[int],
    rng: np.random.Generator,
    max_enumerate: int,
    n_sample: int,
) -> List[Tuple[int, ...]]:
    total = _tuple_product_size(sizes)
    if total <= max_enumerate:
        return [tuple(t) for t in itertools.product(*[range(s) for s in sizes])]
    tuples = {tuple([0] * len(sizes))}
    for _ in range(max(1, n_sample - 1)):
        t = tuple(int(rng.integers(0, s)) for s in sizes)
        tuples.add(t)
        if len(tuples) >= n_sample:
            break
    return list(tuples)


def admissibility_closure(world: WorldInstance, profile: Dict[int, Partition], cfg: SCDCConfig) -> Dict[int, Partition]:
    """Least coarsening to restore induced deterministic quotient maps (admissibility)."""
    rng = np.random.default_rng(cfg.seed)
    new_profile = {v: p.copy() for v, p in profile.items()}

    q = {v: new_profile[v].canonical_label_map() for v in world.G.nodes()}

    for v in world.G.nodes():
        in_edges = world.in_edges[v]
        if not in_edges:
            continue
        in_sizes = [world.alphabet_size[u] for (u, _key) in in_edges]
        tuples = _sample_or_enumerate_tuples(
            in_sizes, rng=rng,
            max_enumerate=cfg.max_enumerate_tuples_per_vertex,
            n_sample=cfg.sample_tuples_per_vertex,
        )

        # induced quotient map must be well-defined:
        # if two raw inputs map to same quotient input class, their outputs must map to same class.
        by_q_in: Dict[Tuple[int, ...], int] = {}
        for t in tuples:
            q_in = tuple(q[in_edges[i][0]][int(t[i])] for i in range(len(in_edges)))
            out = int(world.local_rule[v](tuple(int(x) for x in t)))
            prev = by_q_in.get(q_in)
            if prev is None:
                by_q_in[q_in] = out
            else:
                # outputs must be identified at v
                if q[v][prev] != q[v][out]:
                    new_profile[v].union(prev, out)
                    q[v] = new_profile[v].canonical_label_map()
    return new_profile


def star_equivariance_closure(world: WorldInstance, profile: Dict[int, Partition], cfg: SCDCConfig) -> Dict[int, Partition]:
    """Coarsen profile so that permuting equal-source input wires does not change output class."""
    rng = np.random.default_rng(cfg.seed)
    new_profile = {v: p.copy() for v, p in profile.items()}
    q = {v: new_profile[v].canonical_label_map() for v in world.G.nodes()}

    for v in world.G.nodes():
        in_edges = world.in_edges[v]
        m = len(in_edges)
        if m <= 1:
            continue

        in_sources = [u for (u, _key) in in_edges]
        # For each source with multiplicity > 1, we will test adjacent swaps inside that bundle.
        source_to_positions: Dict[int, List[int]] = {}
        for i, u in enumerate(in_sources):
            source_to_positions.setdefault(u, []).append(i)

        # sample inputs
        in_sizes = [world.alphabet_size[u] for (u, _key) in in_edges]
        tuples = _sample_or_enumerate_tuples(
            in_sizes, rng=rng,
            max_enumerate=cfg.max_enumerate_tuples_per_vertex,
            n_sample=cfg.sample_tuples_per_vertex,
        )

        for src, idxs in source_to_positions.items():
            if len(idxs) <= 1:
                continue
            idxs = sorted(idxs)
            # test adjacent swaps among those positions
            for a, b in zip(idxs[:-1], idxs[1:]):
                for t in tuples:
                    t = list(int(x) for x in t)
                    out1 = int(world.local_rule[v](tuple(t)))
                    t[a], t[b] = t[b], t[a]
                    out2 = int(world.local_rule[v](tuple(t)))
                    if q[v][out1] != q[v][out2]:
                        new_profile[v].union(out1, out2)
                        q[v] = new_profile[v].canonical_label_map()
    return new_profile


def _find_diamonds(cond_dag: nx.DiGraph, depth: Dict[int, int]) -> List[Tuple[int, int, int, int]]:
    """Find diamond patterns x -> y, x -> z, and y -> w, z -> w in condensation DAG."""
    diamonds: List[Tuple[int, int, int, int]] = []
    # small graphs: brute
    for x in cond_dag.nodes():
        succ = list(cond_dag.successors(x))
        for i in range(len(succ)):
            for j in range(i + 1, len(succ)):
                y, z = succ[i], succ[j]
                succ_y = set(cond_dag.successors(y))
                succ_z = set(cond_dag.successors(z))
                for w in succ_y.intersection(succ_z):
                    diamonds.append((x, y, z, w))
    return diamonds


def diamond_confluence_closure(world: WorldInstance, profile: Dict[int, Partition], cfg: SCDCConfig) -> Dict[int, Partition]:
    """Sampling-based closure for diamond confluence.

    We sample states on a neighborhood sufficient to evaluate two SCC update orders across a diamond.
    If the two orders disagree in quotient class for some vertex, merge those output states.
    """
    rng = np.random.default_rng(cfg.seed)
    new_profile = {v: p.copy() for v, p in profile.items()}

    cond = compute_condensation(world.G)
    diamonds = _find_diamonds(cond.dag, cond.depth)
    if not diamonds:
        return new_profile

    node_to_scc = cond.node_to_scc
    sccs = cond.sccs

    def update_scc(state: Dict[int, int], scc_id: int, updated_sccs: set[int]) -> Dict[int, int]:
        old = dict(state)
        cur = dict(state)
        verts = list(sccs[scc_id])
        new_vals: Dict[int, int] = {}
        for v in verts:
            inputs: List[int] = []
            for (u, _key) in world.in_edges[v]:
                su = node_to_scc[u]
                if su in updated_sccs and su != scc_id:
                    inputs.append(cur[u])
                else:
                    inputs.append(old[u])
            new_vals[v] = int(world.local_rule[v](tuple(inputs)))
        for v, val in new_vals.items():
            cur[v] = val
        return cur

    for (x, y, z, _w) in diamonds:
        # neighborhood for SCC x,y,z plus their direct predecessors (to avoid KeyError)
        neigh_nodes: set[int] = set()
        for scc_id in [x, y, z]:
            neigh_nodes |= set(sccs[scc_id])
            for v in sccs[scc_id]:
                for (u, _key) in world.in_edges[v]:
                    neigh_nodes.add(u)

        if not neigh_nodes:
            continue

        for _ in range(cfg.max_diamond_state_samples):
            state = {v: int(rng.integers(0, world.alphabet_size[v])) for v in neigh_nodes}

            updated: set[int] = set()
            s1 = update_scc(state, x, updated_sccs=updated)
            updated.add(x)

            # y then z
            s_yz = update_scc(s1, y, updated_sccs=updated)
            s_yz = update_scc(s_yz, z, updated_sccs=set(updated) | {y})

            # z then y
            s_zy = update_scc(s1, z, updated_sccs=updated)
            s_zy = update_scc(s_zy, y, updated_sccs=set(updated) | {z})

            for scc_id in [y, z]:
                for v in sccs[scc_id]:
                    pv = new_profile[v]
                    qv = pv.canonical_label_map()
                    if qv[s_yz[v]] != qv[s_zy[v]]:
                        pv.union(s_yz[v], s_zy[v])

    return new_profile


def scdc_map(world: WorldInstance, profile: Dict[int, Partition], cfg: SCDCConfig) -> Dict[int, Partition]:
    p1 = admissibility_closure(world, profile, cfg)
    p2 = star_equivariance_closure(world, p1, cfg)
    p3 = diamond_confluence_closure(world, p2, cfg)
    return p3


def compute_lambda_star(world: WorldInstance, cfg: Optional[SCDCConfig] = None) -> Dict[int, Partition]:
    """Compute least fixed point Λ⋆ of the SCDC map above the discrete profile."""
    if cfg is None:
        cfg = SCDCConfig()
    profile = discrete_profile(world)
    for _it in range(cfg.max_iterations):
        new_profile = scdc_map(world, profile, cfg)
        if profile_equal(profile, new_profile):
            return new_profile
        profile = new_profile
    raise RuntimeError(f"SCDC did not converge within {cfg.max_iterations} iterations")


def quotient_loss(world: WorldInstance, profile: Dict[int, Partition]) -> float:
    """Normalized quotient loss: fraction of local alphabet collapsed by Λ⋆.

    0.0 means no identifications needed (already consistent),
    1.0 means everything collapses (maximally inconsistent).

    We use: loss = (Σ_v (|A_v| - |\bar A_v|)) / (Σ_v |A_v|)
    """
    num = 0.0
    den = 0.0
    for v in world.G.nodes():
        Av = int(world.alphabet_size[v])
        den += Av
        num += (Av - int(profile[v].num_classes()))
    return float(num / max(1.0, den))


def quotient_world(
    world: WorldInstance,
    profile: Dict[int, Partition],
    cache_limit: int = 200_000,
    *,
    max_precompute_tuples_per_vertex: int | None = None,
    lazy_cache_limit: int | None = None,
) -> WorldInstance:
    r"""Construct the quotient world \bar W = W / \Lambda with lazy quotient rules."""
    # Backwards-compatibility shims for older experiment scripts.
    if lazy_cache_limit is not None:
        cache_limit = int(lazy_cache_limit)
    # max_precompute_tuples_per_vertex was used by earlier prototypes; it is currently ignored.
    _ = max_precompute_tuples_per_vertex

    q_maps = {v: profile[v].canonical_label_map() for v in world.G.nodes()}

    # representatives for each quotient class (choose min element in class)
    reps_per_vertex: Dict[int, Dict[int, int]] = {}
    for v in world.G.nodes():
        classes = list(profile[v].classes().values())
        classes.sort(key=lambda cls: cls[0] if cls else 10**18)
        reps_per_vertex[v] = {cid: members[0] for cid, members in enumerate(classes)}

    bar_alphabet = {v: int(profile[v].num_classes()) for v in world.G.nodes()}

    bar_rules: Dict[int, LocalRule] = {}
    for v in world.G.nodes():
        in_edges = world.in_edges[v]
        reps_per_coord = [reps_per_vertex[u] for (u, _key) in in_edges]
        q_out_map = q_maps[v]

        raw_rule = world.local_rule[v]
        bar_rules[v] = LazyQuotientRule(
            in_edges=in_edges,
            reps_per_coord=reps_per_coord,
            q_out_map=q_out_map,
            raw_rule=raw_rule,
            cache_limit=cache_limit,
        )

    return WorldInstance(G=world.G, alphabet_size=bar_alphabet, local_rule=bar_rules, in_edges=world.in_edges)


def quotient_state(state: Dict[int, int], profile: Dict[int, Partition]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for v, a in state.items():
        out[v] = int(profile[v].canonical_label_map()[int(a)])
    return out