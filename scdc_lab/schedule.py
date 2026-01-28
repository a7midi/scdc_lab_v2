from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import networkx as nx
import numpy as np

from .graphs import Condensation, compute_condensation
from .world import WorldInstance


def random_topological_order(dag: nx.DiGraph, rng: np.random.Generator) -> List[int]:
    """Random linear extension of a DAG partial order (Kahn with random tie-break)."""
    indeg = {v: int(dag.in_degree(v)) for v in dag.nodes()}
    available = [v for v, d in indeg.items() if d == 0]
    order: List[int] = []
    while available:
        i = int(rng.integers(0, len(available)))
        v = available.pop(i)
        order.append(v)
        for w in dag.successors(v):
            indeg[w] -= 1
            if indeg[w] == 0:
                available.append(w)
    if len(order) != dag.number_of_nodes():
        raise ValueError("DAG topological order failed (graph has a cycle).")
    return order


@dataclass
class ScheduleContext:
    condensation: Condensation

    @classmethod
    def from_world(cls, world: WorldInstance) -> "ScheduleContext":
        return cls(condensation=compute_condensation(world.G))


def tick_update(
    world: WorldInstance,
    state_t: Dict[int, int],
    schedule_scc: Sequence[int],
    condensation: Condensation,
) -> Dict[int, int]:
    """One-tick global update T_σ (sequential SCC update, synchronous within SCC)."""
    old = dict(state_t)
    cur = dict(state_t)

    processed_scc: set[int] = set()
    node_to_scc = condensation.node_to_scc
    sccs = condensation.sccs

    for scc_id in schedule_scc:
        vertices = list(sccs[scc_id])

        new_vals: Dict[int, int] = {}
        for v in vertices:
            inputs: List[int] = []
            for (u, _key) in world.in_edges[v]:
                su = node_to_scc[u]
                if su in processed_scc and su != scc_id:
                    inputs.append(int(cur[u]))
                else:
                    inputs.append(int(old[u]))
            new_vals[v] = int(world.local_rule[v](tuple(inputs)))

        for v, val in new_vals.items():
            cur[v] = int(val)

        processed_scc.add(scc_id)

    return cur


def simulate(
    world: WorldInstance,
    x0: Dict[int, int],
    steps: int,
    seed: Optional[int] = None,
    schedule_per_tick: bool = True,
) -> List[Dict[int, int]]:
    """Simulate for a number of ticks. Returns states including x0."""
    rng = np.random.default_rng(seed)
    ctx = ScheduleContext.from_world(world)
    states = [dict(x0)]
    x = dict(x0)

    fixed_sched: Optional[List[int]] = None
    if not schedule_per_tick:
        fixed_sched = random_topological_order(ctx.condensation.dag, rng)

    for _t in range(int(steps)):
        sched = random_topological_order(ctx.condensation.dag, rng) if schedule_per_tick else fixed_sched
        assert sched is not None
        x = tick_update(world, x, sched, ctx.condensation)
        states.append(dict(x))
    return states


def schedule_independence_check(
    world: WorldInstance,
    x: Dict[int, int],
    n_schedules: int = 8,
    seed: Optional[int] = None,
) -> List[Dict[int, int]]:
    """Compute one-tick results under multiple random admissible schedules."""
    rng = np.random.default_rng(seed)
    cond = compute_condensation(world.G)
    outs = []
    for _ in range(int(n_schedules)):
        sched = random_topological_order(cond.dag, rng)
        outs.append(tick_update(world, x, sched, cond))
    return outs
