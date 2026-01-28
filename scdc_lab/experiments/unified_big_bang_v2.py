from __future__ import annotations

"""A cleaned, library-backed version of the original unified_big_bang_v2 prototype.

It anneals a random directed graph using a motif Hamiltonian (diamonds + feed-forward triangles)
and then runs basic spacetime checks on the resulting graph.

This file exists mainly for continuity with earlier runs; for the full end-to-end pipeline
(spacetime -> matter -> geodesics), use:

    python -m scdc_lab.experiments.unified_consistency_universe --help
"""

import argparse
import time
import numpy as np
import networkx as nx

from ..graphs import er_directed_multigraph
from ..genesis import motif_energy, AnnealConfig, run_genesis, motif_counts
from ..geometry import measure_light_cone_growth
from ..experiments.unified_consistency_universe import plot_spacetime_embedding


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified Big Bang v2 (motif annealing).")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--p", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--w_d", type=float, default=1.0)
    p.add_argument("--w_t", type=float, default=2.0)
    p.add_argument("--temp_start", type=float, default=2.0)
    p.add_argument("--cooling_rate", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_png", type=str, default="big_bang_result.png")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    G0 = er_directed_multigraph(args.n, args.p, seed=args.seed, max_parallel=1)
    a_cfg = AnnealConfig(steps=args.steps, temp_start=args.temp_start, cooling_rate=args.cooling_rate, seed=args.seed)

    print(f"--- GENESIS START (N={args.n}) ---")
    t0 = time.time()
    energy_fn = lambda H: motif_energy(H, w_d=args.w_d, w_t=args.w_t)
    G, hist = run_genesis(G0, energy_fn=energy_fn, cfg=a_cfg, progress_every=500)
    print(f"--- GENESIS COMPLETE in {time.time()-t0:.2f}s ---")

    c = motif_counts(G)
    print(f"[Counts] Diamonds={int(c.diamonds)}  FF_Triangles={int(c.ff_triangles)}")

    growth = measure_light_cone_growth(G, t_max=8, n_sources=6, seed=args.seed)
    models = [g.growth_model for g in growth]
    print(f"[Time Check] Light cone models: {models}")
    eff_dims = [g.eff_dim for g in growth if g.eff_dim is not None]
    if eff_dims:
        print(f"[Time Check] Effective dim median (power-law): {float(np.median(eff_dims)):.2f}")

    plot_spacetime_embedding(G, out_png=args.out_png, title="Emergent Causal Geodesics (Condensation MDS)")
    print(f">> Snapshot saved to {args.out_png}")


if __name__ == "__main__":
    main()
