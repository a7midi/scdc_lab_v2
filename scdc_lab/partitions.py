from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Partition:
    """Union-Find partition for a finite set {0,1,...,n-1}.

    We use this to represent local congruences (equivalence relations) on finite alphabets.
    """

    n: int
    parent: List[int]
    rank: List[int]

    @classmethod
    def discrete(cls, n: int) -> "Partition":
        return cls(n=n, parent=list(range(n)), rank=[0] * n)

    def copy(self) -> "Partition":
        return Partition(self.n, self.parent.copy(), self.rank.copy())

    def find(self, a: int) -> int:
        if self.parent[a] != a:
            self.parent[a] = self.find(self.parent[a])
        return self.parent[a]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def classes(self) -> Dict[int, List[int]]:
        d: Dict[int, List[int]] = {}
        for a in range(self.n):
            r = self.find(a)
            d.setdefault(r, []).append(a)
        # Keep class members sorted for stable downstream behavior
        for r in d:
            d[r].sort()
        return d

    def num_classes(self) -> int:
        return len(self.classes())

    def canonical_label_map(self) -> Dict[int, int]:
        """Return a stable map element -> class_id in {0..k-1}.

        Important:
        - Union-Find representatives (roots) depend on union order.
        - Therefore, we *cannot* canonically label classes by sorting root ids.
        - We instead label classes by sorting them by their *minimum element*.
        This makes equality/FP checks stable.
        """
        classes = list(self.classes().values())
        classes.sort(key=lambda cls: cls[0] if cls else 10**18)
        elem_to_id: Dict[int, int] = {}
        for cid, members in enumerate(classes):
            for a in members:
                elem_to_id[a] = cid
        return elem_to_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Partition) or self.n != other.n:
            return False
        return self.canonical_label_map() == other.canonical_label_map()


def profile_equal(profile_a: Dict[int, Partition], profile_b: Dict[int, Partition]) -> bool:
    if profile_a.keys() != profile_b.keys():
        return False
    return all(profile_a[v] == profile_b[v] for v in profile_a)
