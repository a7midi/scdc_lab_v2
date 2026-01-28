from __future__ import annotations

"""Relativistic Big Bang v3 (library-backed).

This script focuses on:
- Light cone expansion (locality / causal spacetime)
- Curvature proxy field on condensation DAG
- A visualization of the condensation embedding

For an end-to-end spacetime->matter->geodesics run, use unified_consistency_universe.
"""

import argparse
import numpy as np
import networkx as nx

from ..graphs import er_directed_multigraph
from ..genesis import motif_energy, AnnealConfig, run_genesis
from ..geometry import measure_light_cone_growth, curvature_field_on_condensation
from .unified_consistency_universe import plot_spacetime_embedding


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified Big Bang v3 (relativistic diagnostics).")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--p", type=float, default=0.02)
    p.add_argument("--genesis_steps", type=int, default=14000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_png", type=str, default="relativistic_universe.png")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    rng = np.random.default_rng(args.seed)

    G0 = er_directed_multigraph(args.n, args.p, seed=args.seed, max_parallel=1)

    if args.genesis_steps > 0:
        print(f"--- RELATIVISTIC GENESIS (N={args.n}) ---")
        a_cfg = AnnealConfig(steps=args.genesis_steps, temp_start=2.0, cooling_rate=0.999, seed=args.seed)
        energy_fn = lambda H: motif_energy(H, w_d=1.0, w_t=2.0)
        G, _hist = run_genesis(G0, energy_fn=energy_fn, cfg=a_cfg, progress_every=1000)
    else:
        G = G0

    # GR Test 1: Light cone expansion
    print("\n[GR Test 1] Measuring Light Cone Expansion...")
    growth = measure_light_cone_growth(G, t_max=8, n_sources=8, seed=args.seed)
    vols = growth[0].volumes if growth else []
    print(f"Light Cone Volumes by T: {vols}")
    models = [g.growth_model for g in growth]
    if models.count("polynomial") >= max(1, len(models)//2):
        print(">> Expansion: Polynomial (Geometric / Local Spacetime)")
    elif models.count("exponential") >= max(1, len(models)//2):
        print(">> Expansion: Exponential (Nonlocal / Expander-like)")
    else:
        print(">> Expansion: Mixed/Unclear")

    # GR Test 2: Curvature field proxy
    print("\n[GR Test 2] Measuring Discrete Curvature Field...")
    cf = curvature_field_on_condensation(G, r_cone=1)
    curv_vals = np.array(list(cf.kappa_node.values()), dtype=float) if cf.kappa_node else np.array([0.0])
    mean_curv = float(np.mean(curv_vals))
    var_curv = float(np.var(curv_vals))
    print(f"Mean Curvature Proxy (kappa_node): {mean_curv:.3f}")
    print(f"Curvature Variance: {var_curv:.4f}")
    if var_curv < 1e-3:
        print(">> Spacetime is Smooth/Flat-ish")
    else:
        print(">> Spacetime is Rugged/Singular")

    # Visualize
    print("\n[Visual] Rendering Spacetime Manifold...")
    plot_spacetime_embedding(G, out_png=args.out_png, title="Relativistic Universe (Condensation MDS)")
    print(f">> Saved '{args.out_png}'")


if __name__ == "__main__":
    main()
