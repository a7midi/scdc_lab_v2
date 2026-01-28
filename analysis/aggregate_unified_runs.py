#!/usr/bin/env python3
"""
Aggregate summary JSONs produced by scdc_lab/experiments/unified_consistency_universe.py.

This script is intentionally conservative: it only uses fields already written
to the summary JSON. It does NOT re-run simulations.

It outputs:
- results/run_level_transport.csv
- results/pf_level_transport.csv
- results/fig_transport_regimes.png
- results/fig_volume_spectrum_pf012_pf014.png (if pf 0.12 and 0.14 exist)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PF_SEED_RE = re.compile(r"pf(?P<pf>[0-9.]+)_seed(?P<seed>\d+)_summary\.json$")


def parse_pf_seed(path: Path) -> Tuple[float, int]:
    m = PF_SEED_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not parse pf/seed from filename: {path.name}")
    return float(m.group("pf")), int(m.group("seed"))


def classify_volume(v: int, dead_lt: int, loc_le: int) -> str:
    # Match the operational definitions used in the PRL draft.
    # Dead: v < dead_lt
    # Localized: dead_lt <= v <= loc_le
    # Shockwave: v > loc_le
    if v < dead_lt:
        return "dead"
    if v <= loc_le:
        return "localized"
    return "shockwave"


def max_consecutive_gap(values: List[int]) -> Tuple[int, int, int]:
    """Return (from, to, gap_size) for the largest gap between consecutive unique integers."""
    uniq = sorted(set(values))
    if len(uniq) < 2:
        return (uniq[0] if uniq else -1, uniq[0] if uniq else -1, 0)
    gaps = [(uniq[i], uniq[i + 1], uniq[i + 1] - uniq[i]) for i in range(len(uniq) - 1)]
    a, b, g = max(gaps, key=lambda t: t[2])
    return a, b, g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs", help="Directory containing *_summary.json files.")
    ap.add_argument("--pattern", type=str, default="pf*_seed*_summary.json", help="Glob pattern within runs_dir.")
    ap.add_argument("--out_dir", type=str, default="results", help="Where to write CSVs/figures.")
    ap.add_argument("--dead_lt", type=int, default=5, help="Dead if v_last < dead_lt.")
    ap.add_argument("--loc_le", type=int, default=20, help="Localized if dead_lt <= v_last <= loc_le.")
    ap.add_argument("--t_horizon", type=int, default=8, help="Which light-cone tick to use as v_last (index).")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(runs_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files matched {runs_dir}/{args.pattern}")

    rows: List[Dict[str, Any]] = []
    lc_rows: List[Dict[str, Any]] = []

    for p in paths:
        pf, seed = parse_pf_seed(p)
        d = json.loads(p.read_text())

        # light cones
        lcs = d.get("light_cones", [])
        v_last_list: List[int] = []
        eff_dims: List[float] = []
        poly_ct = 0
        exp_ct = 0

        for lc in lcs:
            vols = lc.get("volumes", [])
            if not vols:
                continue
            # if t_horizon exceeds available length, use the last element
            if args.t_horizon < len(vols):
                v_last = int(vols[args.t_horizon])
            else:
                v_last = int(vols[-1])
            v_last_list.append(v_last)

            eff = lc.get("eff_dim", None)
            if eff is not None:
                eff_dims.append(float(eff))

            gm = lc.get("growth_model", "")
            if gm == "poly":
                poly_ct += 1
            elif gm == "exp":
                exp_ct += 1

            lc_rows.append(
                {
                    "pf": pf,
                    "seed": seed,
                    "file": p.name,
                    "source": lc.get("source", None),
                    "v_last": v_last,
                    "eff_dim": lc.get("eff_dim", None),
                    "growth_model": gm,
                    "status": classify_volume(v_last, args.dead_lt, args.loc_le),
                }
            )

        # classify volumes for run-level summary
        status_counts = {"dead": 0, "localized": 0, "shockwave": 0}
        for v in v_last_list:
            status_counts[classify_volume(v, args.dead_lt, args.loc_le)] += 1

        tot = max(1, sum(status_counts.values()))
        gap_from, gap_to, gap_size = max_consecutive_gap(v_last_list)

        lens = d.get("lensing", {}) or {}
        pockets = d.get("pockets", {}) or {}

        rows.append(
            {
                "pf": pf,
                "seed": seed,
                "file": p.name,
                "N": d.get("N", None),
                "E": d.get("E", None),
                "scdc_loss": d.get("scdc_loss", None),
                "spectral_gap_sym": d.get("spectral_gap_sym", None),
                "mean_min_distance": lens.get("mean_min_distance", None),
                "pocket_best_size": pockets.get("best_size", None),
                "pocket_best_persistence": pockets.get("best_persistence", None),
                "n_lightcones": len(v_last_list),
                "dead_ct": status_counts["dead"],
                "localized_ct": status_counts["localized"],
                "shockwave_ct": status_counts["shockwave"],
                "dead_frac": status_counts["dead"] / tot,
                "localized_frac": status_counts["localized"] / tot,
                "shockwave_frac": status_counts["shockwave"] / tot,
                "eff_dim_median": float(np.median(eff_dims)) if eff_dims else None,
                "growth_poly_frac": poly_ct / max(1, poly_ct + exp_ct),
                "max_gap": gap_size,
                "gap_from": gap_from,
                "gap_to": gap_to,
            }
        )

    run_df = pd.DataFrame(rows).sort_values(["pf", "seed"])
    lc_df = pd.DataFrame(lc_rows).sort_values(["pf", "seed", "source"])

    run_csv = out_dir / "run_level_transport.csv"
    lc_csv = out_dir / "lightcone_level_transport.csv"
    run_df.to_csv(run_csv, index=False)
    lc_df.to_csv(lc_csv, index=False)

    # Aggregate by pf
    agg = lc_df.groupby(["pf", "status"]).size().unstack(fill_value=0)
    agg["total"] = agg.sum(axis=1)
    for col in ["dead", "localized", "shockwave"]:
        if col not in agg.columns:
            agg[col] = 0
    agg_pct = agg[["dead", "localized", "shockwave"]].div(agg["total"], axis=0)
    pf_df = agg_pct.reset_index().sort_values("pf")
    pf_csv = out_dir / "pf_level_transport.csv"
    pf_df.to_csv(pf_csv, index=False)

    # Figure: regime fractions vs pf
    fig1 = plt.figure()
    x = pf_df["pf"].values
    plt.plot(x, pf_df["dead"].values, marker="o", label="dead (v<dead_lt)")
    plt.plot(x, pf_df["localized"].values, marker="o", label="localized")
    plt.plot(x, pf_df["shockwave"].values, marker="o", label="shockwave (v>loc_le)")
    plt.xlabel("p_forward")
    plt.ylabel("fraction of sources")
    plt.legend()
    plt.title("Transport regimes vs density (horizon-limited light-cone volume)")
    fig1.tight_layout()
    fig1_path = out_dir / "fig_transport_regimes.png"
    fig1.savefig(fig1_path, dpi=200)
    plt.close(fig1)

    # Figure: volume spectrum for pf=0.12 and pf=0.14 if present
    have_pf = set(lc_df["pf"].unique().tolist())
    if 0.12 in have_pf and 0.14 in have_pf:
        fig2 = plt.figure()
        for pf in [0.12, 0.14]:
            vals = lc_df.loc[lc_df["pf"] == pf, "v_last"].astype(int).values
            # jitter x a tiny bit for visibility
            jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.01
            plt.scatter(np.full(len(vals), pf) + jitter, vals, s=18, alpha=0.8, label=f"pf={pf}")
        plt.yscale("log")
        plt.xlabel("p_forward")
        plt.ylabel("v_last at horizon (log scale)")
        plt.title("Horizon-limited volume spectrum (pf=0.12 vs pf=0.14)")
        plt.legend()
        fig2.tight_layout()
        fig2_path = out_dir / "fig_volume_spectrum_pf012_pf014.png"
        fig2.savefig(fig2_path, dpi=200)
        plt.close(fig2)

    print(f"Wrote:\n  {run_csv}\n  {lc_csv}\n  {pf_csv}\n  {fig1_path}")
    if 0.12 in have_pf and 0.14 in have_pf:
        print(f"  {out_dir / 'fig_volume_spectrum_pf012_pf014.png'}")


if __name__ == "__main__":
    main()
