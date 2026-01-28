#!/usr/bin/env python3
"""
H1 CSV Analyzer

This script is tolerant of two CSV schemas:

Schema A (glider_search / pocket-tracking):
    t, active_size, active_centroid_depth,
    pocket_size, pocket_centroid_depth,
    jaccard_prev, active_jaccard_prev

Schema B (radiation_search / detector-tracking):
    t, active_size, active_centroid_depth,
    active_max_depth, active_p95_depth,
    detector_count, detector_ge_count,
    active_jaccard_prev
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _linreg_slope(x: Sequence[float], y: Sequence[float]) -> float:
    """Return slope of y ~ a + b x. If ill-posed, return 0."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return 0.0
    if np.allclose(x, x[0]):
        return 0.0
    b = np.polyfit(x, y, 1)[0]
    return float(b)


def _series_stats(a: pd.Series) -> tuple[float, float, float]:
    a2 = pd.to_numeric(a, errors="coerce")
    return float(a2.min()), float(a2.median()), float(a2.max())


def _jaccard_stats(a: pd.Series) -> tuple[float, float, float]:
    a2 = pd.to_numeric(a, errors="coerce").dropna()
    if len(a2) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(a2.median()), float(a2.min()), float(a2.iloc[-1])


@dataclass
class Verdict:
    label: str
    reason: str


def _localization_verdict(frac_median: float, frac_max: float) -> Verdict:
    if math.isfinite(frac_median) and frac_median >= 0.60:
        return Verdict("Percolated", f"median fraction={frac_median:.3f} >= 0.600")
    if math.isfinite(frac_max) and frac_max <= 0.30:
        return Verdict("Localized", f"max fraction={frac_max:.3f} <= 0.300")
    return Verdict("Ambiguous", f"median={frac_median:.3f}, max={frac_max:.3f}")


def analyze_glider(df: pd.DataFrame, burn: int, n_total: Optional[int]) -> None:
    print("=== H1 CSV Analysis (pocket tracking) ===")
    print(f"Rows: {len(df)}  (ticks {int(df['t'].min())}..{int(df['t'].max())})")

    pmin, pmed, pmax = _series_stats(df["pocket_size"])
    amin, amed, amax = _series_stats(df["active_size"])
    print(f"Pocket size: min={pmin:.0f}  median={pmed:.1f}  max={pmax:.0f}")
    print(f"Active size:  min={amin:.0f}  median={amed:.1f}  max={amax:.0f}")

    if n_total is not None:
        pocket_frac = df["pocket_size"] / float(n_total)
        active_frac = df["active_size"] / float(n_total)
        pf_med = float(pd.to_numeric(pocket_frac, errors="coerce").median())
        pf_max = float(pd.to_numeric(pocket_frac, errors="coerce").max())
        af_med = float(pd.to_numeric(active_frac, errors="coerce").median())
        af_max = float(pd.to_numeric(active_frac, errors="coerce").max())
        print(f"n_total: {n_total}")
        print(f"Pocket fraction: median={pf_med:.3f} max={pf_max:.3f}")
        print(f"Active fraction: median={af_med:.3f} max={af_max:.3f}")

    # Stabilization tick
    thr = 0.95 * pmax
    stab = df.loc[df["pocket_size"] >= thr, "t"]
    stab_tick = int(stab.iloc[0]) if len(stab) else None
    print(f"Stabilization tick (>= 0.95*max): {stab_tick}")

    # Centroid drift
    print(
        f"Pocket centroid depth: start={float(df['pocket_centroid_depth'].iloc[0]):.3f}  "
        f"end={float(df['pocket_centroid_depth'].iloc[-1]):.3f}"
    )
    df_burn = df[df["t"] >= burn].copy()
    pocket_slope = _linreg_slope(df_burn["t"], df_burn["pocket_centroid_depth"])
    active_slope = _linreg_slope(df_burn["t"], df_burn["active_centroid_depth"])
    print(
        f"Drift slope after burn={burn}: pocket_slope={pocket_slope:.6g} depth/tick  "
        f"active_slope={active_slope:.6g} depth/tick"
    )

    # Jaccard
    j_med, j_min, j_last = _jaccard_stats(df.get("jaccard_prev", pd.Series(dtype=float)))
    aj_med, aj_min, aj_last = _jaccard_stats(df.get("active_jaccard_prev", pd.Series(dtype=float)))
    print(f"Jaccard(prev): pocket median={j_med:.3f} min={j_min:.3f} last={j_last:.3f}")
    print(f"Jaccard(prev): active  median={aj_med:.3f} min={aj_min:.3f} last={aj_last:.3f}")

    if len(df_burn) > 0:
        j2_med, j2_min, _ = _jaccard_stats(df_burn.get("jaccard_prev", pd.Series(dtype=float)))
        aj2_med, aj2_min, _ = _jaccard_stats(df_burn.get("active_jaccard_prev", pd.Series(dtype=float)))
        print(f"Jaccard(prev) after burn: pocket median={j2_med:.3f} min={j2_min:.3f}")
        print(f"Jaccard(prev) after burn: active  median={aj2_med:.3f} min={aj2_min:.3f}")

    if n_total is not None:
        pocket_frac = df["pocket_size"] / float(n_total)
        vf = _localization_verdict(float(pocket_frac.median()), float(pocket_frac.max()))
        print(f"Verdict: {vf.label}. {vf.reason}")
    else:
        print("Verdict: (skipped) Provide --n_total for a localization verdict.")


def analyze_radiation(df: pd.DataFrame, burn: int, n_total: Optional[int]) -> None:
    print("=== H1 CSV Analysis (radiation / detector) ===")
    print(f"Rows: {len(df)}  (ticks {int(df['t'].min())}..{int(df['t'].max())})")

    amin, amed, amax = _series_stats(df["active_size"])
    print(f"Active size: min={amin:.0f}  median={amed:.1f}  max={amax:.0f}")

    if n_total is not None:
        active_frac = df["active_size"] / float(n_total)
        af_med = float(pd.to_numeric(active_frac, errors="coerce").median())
        af_max = float(pd.to_numeric(active_frac, errors="coerce").max())
        print(f"n_total: {n_total}")
        print(f"Active fraction: median={af_med:.3f} max={af_max:.3f}")
        vf = _localization_verdict(af_med, af_max)
        print(f"Verdict: {vf.label}. {vf.reason}")

    # Depth / propagation
    if "active_max_depth" in df.columns:
        dmax = int(pd.to_numeric(df["active_max_depth"], errors="coerce").max())
        print(f"Max active depth seen: {dmax}")
    if "active_p95_depth" in df.columns:
        p95_end = float(pd.to_numeric(df["active_p95_depth"], errors="coerce").iloc[-1])
        print(f"Active p95 depth (final): {p95_end:.3f}")

    # Detector
    if "detector_count" in df.columns:
        hit = df.index[pd.to_numeric(df["detector_count"], errors="coerce").fillna(0) > 0]
        first_hit = int(df.loc[hit[0], "t"]) if len(hit) else None
        total_hits = int(pd.to_numeric(df["detector_count"], errors="coerce").fillna(0).sum())
        print(f"Detector: first_hit_tick={first_hit}  total_hits={total_hits}")

    # Drift
    df_burn = df[df["t"] >= burn].copy()
    active_slope = _linreg_slope(df_burn["t"], df_burn["active_centroid_depth"])
    print(f"Drift slope after burn={burn}: active_slope={active_slope:.6g} depth/tick")

    # Jaccard
    aj_med, aj_min, aj_last = _jaccard_stats(df.get("active_jaccard_prev", pd.Series(dtype=float)))
    print(f"Jaccard(prev): active median={aj_med:.3f} min={aj_min:.3f} last={aj_last:.3f}")
    if len(df_burn) > 0:
        aj2_med, aj2_min, _ = _jaccard_stats(df_burn.get("active_jaccard_prev", pd.Series(dtype=float)))
        print(f"Jaccard(prev) after burn: active median={aj2_med:.3f} min={aj2_min:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="CSV produced by h1_glider_search or h1_radiation_search")
    ap.add_argument("--burn", type=int, default=20, help="burn-in ticks to ignore for drift/jaccard summaries")
    ap.add_argument("--n_total", type=int, default=None, help="total number of nodes (enables fraction-based verdicts)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    if "t" not in df.columns:
        raise SystemExit("CSV missing required column: t")

    if "pocket_size" in df.columns and "pocket_centroid_depth" in df.columns:
        analyze_glider(df, burn=int(args.burn), n_total=args.n_total)
        return
    if "active_max_depth" in df.columns or "detector_count" in df.columns:
        analyze_radiation(df, burn=int(args.burn), n_total=args.n_total)
        return

    raise SystemExit(f"Unrecognized CSV schema. Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
