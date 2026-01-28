from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import itertools
import numpy as np
import networkx as nx


InputTuple = Tuple[int, ...]


class LocalRule:
    """Abstract local update rule."""

    def __call__(self, inputs: InputTuple) -> int:  # pragma: no cover
        raise NotImplementedError

    def table(self) -> Optional[Dict[InputTuple, int]]:
        return None


@dataclass
class LookupTableRule(LocalRule):
    """Deterministic rule specified by an explicit lookup table."""

    table_dict: Dict[InputTuple, int]

    def __call__(self, inputs: InputTuple) -> int:
        return int(self.table_dict[tuple(int(x) for x in inputs)])

    def table(self) -> Dict[InputTuple, int]:
        return self.table_dict


@dataclass
class ThresholdRule(LocalRule):
    """Binary threshold rule: output 1 if sum(inputs) >= threshold else 0."""

    threshold: int
    vacuum_zero: bool = True

    def __call__(self, inputs: InputTuple) -> int:
        s = int(sum(int(x) for x in inputs))
        if self.vacuum_zero and s == 0:
            return 0
        return 1 if s >= int(self.threshold) else 0


@dataclass
class XorRule(LocalRule):
    """Binary XOR of all inputs."""

    def __call__(self, inputs: InputTuple) -> int:
        x = 0
        for a in inputs:
            x ^= (int(a) & 1)
        return int(x)


def random_lookup_rule(
    in_sizes: Sequence[int],
    out_size: int,
    rng: np.random.Generator,
    vacuum_fixed: Optional[int] = 0,
) -> LookupTableRule:
    """Create a random lookup-table rule.

    If vacuum_fixed is not None, force output on the all-zero input tuple to be vacuum_fixed.
    """
    table: Dict[InputTuple, int] = {}
    for inputs in itertools.product(*[range(s) for s in in_sizes]):
        table[tuple(inputs)] = int(rng.integers(0, out_size))
    if vacuum_fixed is not None:
        table[tuple([0] * len(in_sizes))] = int(vacuum_fixed)
    return LookupTableRule(table)


RuleFactory = Callable[[int, Sequence[int], np.random.Generator], LocalRule]


def make_rule_factory(
    rule: str,
    *,
    threshold: int = 1,
    alphabet_k: int = 2,
    vacuum_fixed: int = 0,
) -> RuleFactory:
    """Factory producing per-vertex LocalRule objects.

    Parameters
    ----------
    rule:
        'threshold', 'xor', 'random'
    """
    rule = str(rule).lower().strip()
    if rule == "threshold":
        def factory(_v: int, _in_sizes: Sequence[int], _rng: np.random.Generator) -> LocalRule:
            return ThresholdRule(threshold=int(threshold), vacuum_zero=True)
        return factory

    if rule == "xor":
        def factory(_v: int, _in_sizes: Sequence[int], _rng: np.random.Generator) -> LocalRule:
            return XorRule()
        return factory

    if rule == "random":
        def factory(_v: int, in_sizes: Sequence[int], rng: np.random.Generator) -> LocalRule:
            return random_lookup_rule(in_sizes, out_size=int(alphabet_k), rng=rng, vacuum_fixed=int(vacuum_fixed))
        return factory

    raise ValueError(f"Unknown rule: {rule}")


@dataclass
class WorldInstance:
    """Finite relational world instance (directed multigraph + alphabets + deterministic local rules)."""

    G: nx.MultiDiGraph
    alphabet_size: Dict[int, int]                 # |A_v|
    local_rule: Dict[int, LocalRule]              # λ_v
    in_edges: Dict[int, List[Tuple[int, int]]]    # stable list of (source, key) for each v

    @classmethod
    def homogeneous(
        cls,
        G: nx.MultiDiGraph,
        k: int,
        rule_factory: RuleFactory,
        seed: Optional[int] = None,
    ) -> "WorldInstance":
        rng = np.random.default_rng(seed)
        alphabet_size = {v: int(k) for v in G.nodes()}
        in_edges: Dict[int, List[Tuple[int, int]]] = {}
        local_rule: Dict[int, LocalRule] = {}
        for v in G.nodes():
            inc = [(u, key) for (u, _v, key) in G.in_edges(v, keys=True)]
            inc.sort()
            in_edges[v] = inc
            in_sizes = [alphabet_size[u] for (u, _key) in inc]
            local_rule[v] = rule_factory(v, in_sizes, rng)
        return cls(G=G, alphabet_size=alphabet_size, local_rule=local_rule, in_edges=in_edges)

    def vacuum_state(self, value: int = 0) -> Dict[int, int]:
        return {v: int(value) for v in self.G.nodes()}

    def random_state(self, seed: Optional[int] = None) -> Dict[int, int]:
        rng = np.random.default_rng(seed)
        return {v: int(rng.integers(0, self.alphabet_size[v])) for v in self.G.nodes()}

    def eval_local(self, v: int, source_state: Dict[int, int]) -> int:
        inputs: List[int] = []
        for (u, _key) in self.in_edges[v]:
            inputs.append(int(source_state[u]))
        return int(self.local_rule[v](tuple(inputs)))
