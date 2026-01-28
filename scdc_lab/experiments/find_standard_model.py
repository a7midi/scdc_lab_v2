from __future__ import annotations

"""Exploratory scanner: look for 'Standard Model' sized finite groups inside braid images mod p.

This is *not* a proof that SU(2) or SU(3) emerges — it is a computational probe of whether
the local 'consistency group' (modeled here by braid-group images) naturally hits the
finite group orders associated with SL/PSL families.

For spacetime/matter/geodesics from SCDC, use unified_consistency_universe instead.
"""

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..symmetry import (
    reduced_burau_B3_mod_p,
    burau_like_B4_mod_p,
    matrix_group_generated,
    is_abelian,
)


def order_GL(n: int, p: int) -> int:
    p = int(p)
    n = int(n)
    out = 1
    for k in range(n):
        out *= (p**n - p**k)
    return int(out)


def order_SL(n: int, p: int) -> int:
    return int(order_GL(n, p) // (p - 1))


def order_PSL(n: int, p: int) -> int:
    sl = order_SL(n, p)
    return int(sl // math.gcd(n, p - 1))


@dataclass
class ScanResult:
    n_strands: int
    dim: int
    p: int
    t: int
    order: Optional[int]
    truncated: bool
    nonabelian: bool
    label: str


def scan_sector(
    n_strands: int,
    primes: Sequence[int],
    t_values: Sequence[int],
    max_group_size: int = 20000,
) -> List[ScanResult]:
    results: List[ScanResult] = []
    dim = n_strands - 1
    for p in primes:
        p = int(p)
        if p < 2:
            continue
        targets: Dict[str, int] = {}
        if dim == 2:
            targets["SL(2)"] = order_SL(2, p)
            targets["GL(2)"] = order_GL(2, p)
        if dim == 3:
            targets["SL(3)"] = order_SL(3, p)
            targets["PSL(3)"] = order_PSL(3, p)

        print(f"--- Searching N={n_strands} (dim {dim}) over F_{p} ---")
        print(f"Targets: {targets}")

        for t in t_values:
            t = int(t)
            if (t % p) == 0:
                continue

            if n_strands == 3:
                g1, g2 = reduced_burau_B3_mod_p(t=t, p=p)
                gens = [g1, g2]
            elif n_strands == 4:
                g1, g2, g3 = burau_like_B4_mod_p(t=t, p=p)
                gens = [g1, g2, g3]
            else:
                raise ValueError("Only N=3 or N=4 supported in this scanner.")

            grp = matrix_group_generated(gens, p=p, max_size=int(max_group_size))
            nonab = (not is_abelian(grp.elements, p=p))
            order = grp.order

            label = "Unknown"
            if order is not None:
                for name, size in targets.items():
                    if order == size:
                        label = f"MATCH {name}"
                        break
                if label == "Unknown":
                    # containment heuristics
                    for name, size in targets.items():
                        if order != 0 and size % order == 0:
                            label = f"Subgroup of {name}"
                            break
                        if size != 0 and order % size == 0:
                            label = f"Supergroup of {name}"
                            break
            else:
                label = "Truncated"

            print(f"[p={p}, t={t}] Group order: {order} ({label}) truncated={grp.truncated} nonabelian={nonab}")
            results.append(ScanResult(n_strands=n_strands, dim=dim, p=p, t=t, order=order, truncated=grp.truncated, nonabelian=nonab, label=label))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Find Standard Model candidates via braid images mod p (exploratory).")
    parser.add_argument("--search_weak", action="store_true", help="Scan N=3 (2x2) sector.")
    parser.add_argument("--search_strong", action="store_true", help="Scan N=4 (3x3) sector.")
    parser.add_argument("--primes", type=int, nargs="*", default=[2, 3, 5, 7, 11, 13])
    parser.add_argument("--t", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--max_group_size", type=int, default=20000)
    args = parser.parse_args()

    if not args.search_weak and not args.search_strong:
        print("No mode selected. Running demo scan for both.")
        args.search_weak = True
        args.search_strong = True

    if args.search_weak:
        print("\n=== WEAK FORCE SECTOR (N=3 strands) ===")
        scan_sector(3, primes=args.primes, t_values=args.t, max_group_size=args.max_group_size)

    if args.search_strong:
        print("\n=== STRONG FORCE SECTOR (N=4 strands) ===")
        scan_sector(4, primes=args.primes, t_values=args.t, max_group_size=args.max_group_size)


if __name__ == "__main__":
    main()
