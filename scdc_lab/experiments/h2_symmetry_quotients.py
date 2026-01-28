from __future__ import annotations

"""Hypothesis 2: Symmetry quotients.

This script reports:
- The star (input-wire) symmetry group size, interpreted as AutIn(v).
- The size of the braid-group image modulo p for a Burau-like representation.

This is an exploratory bridge between local consistency symmetries and finite group structure.
"""

import argparse
import math

from ..symmetry import (
    reduced_burau_B3_mod_p,
    burau_like_B4_mod_p,
    matrix_group_generated,
    is_abelian,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="H2: Star symmetries + braid images mod p.")
    p.add_argument("--n_wires", type=int, default=3, help="Number of strands/wires (3 -> 2x2, 4 -> 3x3).")
    p.add_argument("--p", type=int, default=7, help="Prime modulus.")
    p.add_argument("--t", type=int, default=2, help="Burau parameter t (mod p).")
    p.add_argument("--max_group_size", type=int, default=20000)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    n = int(args.n_wires)
    p = int(args.p)
    t = int(args.t)

    # star group: permutations of n identical wires => S_n
    star_group_order = math.factorial(n)
    star_group_generators = max(0, n - 1)

    # braid image
    if n == 3:
        g1, g2 = reduced_burau_B3_mod_p(t=t, p=p)
        gens = [g1, g2]
    elif n == 4:
        g1, g2, g3 = burau_like_B4_mod_p(t=t, p=p)
        gens = [g1, g2, g3]
    else:
        raise ValueError("Only n_wires in {3,4} supported.")

    grp = matrix_group_generated(gens, p=p, max_size=int(args.max_group_size))
    nonabel = not is_abelian(grp.elements, p=p)

    print({
        "star_group_order": int(star_group_order),
        "star_group_generators": int(star_group_generators),
        "braid_group_order_mod_p": grp.order,
        "braid_group_truncated": bool(grp.truncated),
        "braid_group_nonabelian": bool(nonabel),
        "p": int(p),
        "t": int(t),
        "max_group_size": int(args.max_group_size),
    })


if __name__ == "__main__":
    main()
