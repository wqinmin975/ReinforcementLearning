# -*- coding: utf-8 -*-
"""Capture the ACTUAL axis limits used by plot_all_figures (before and after the fix).

Usage: capture_limits.py <outdir>
Monkeypatches Figure.savefig so the real limits of the SAVED figures are recorded,
then compares them against every data point +/- its error bar.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure as mfig
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:/ace/语义UAV")
import uav_semantic_multi_seed_ablation as m

OUTDIR = sys.argv[1]
os.makedirs(OUTDIR, exist_ok=True)
m.SAVE_DIR = OUTDIR

# --- capture the figures at the moment they are written to disk -------------
captured = []
_orig_savefig = mfig.Figure.savefig


def _spy(self, fname, *args, **kwargs):
    res = _orig_savefig(self, fname, *args, **kwargs)
    entry = {"file": os.path.basename(str(fname)),
             "size_in": tuple(round(v, 3) for v in self.get_size_inches()),
             "dpi": kwargs.get("dpi", self.dpi),
             "axes": []}
    for ax in self.axes:
        entry["axes"].append({"xlim": ax.get_xlim(), "ylim": ax.get_ylim()})
    captured.append(entry)
    return res


mfig.Figure.savefig = _spy

df = pd.read_csv(r"D:/ace/语义UAV/exp_results/ablation_multi_seed.csv")
m.plot_all_figures(df)
mfig.Figure.savefig = _orig_savefig

# --- data used by the figures ----------------------------------------------
fix_sigma = m.sigma_est_list[len(m.sigma_est_list) // 2]
fix_rho = m.rho_list[len(m.rho_list) // 2]
SCHEMES = ["PPO", "DQN", "Fixed", "EqualGran"]
df_s = df[np.isclose(df["sigma_est"], fix_sigma)].sort_values("rho")
df_r = df[np.isclose(df["rho"], fix_rho)].sort_values("sigma_est")
df_p = df[np.isclose(df["rho"], fix_rho)].sort_values("sigma_est")


def check_ylim(name, ax, sub, xcol):
    xlo, xhi = ax["xlim"]
    ylo_lim, yhi_lim = ax["ylim"]
    print(f"\n  [{name}]  xlim=({xlo:.6g}, {xhi:.6g})   ylim=({ylo_lim:.6g}, {yhi_lim:.6g})")
    worst_lo, worst_hi = np.inf, -np.inf
    npts = 0
    for p in SCHEMES:
        for x, y, e in zip(sub[xcol], sub[f"{p}_recall_mean"], sub[f"{p}_recall_std"]):
            npts += 1
            lo, hi = y - e, y + e
            worst_lo = min(worst_lo, lo)
            worst_hi = max(worst_hi, hi)
            bad = []
            if lo < ylo_lim:
                bad.append(f"LOW by {ylo_lim - lo:.5f}")
            if hi > yhi_lim:
                bad.append(f"HIGH by {hi - yhi_lim:.5f}")
            if bad:
                print(f"      CLIPPED {p:<10} x={x:.3g} y={y:.4f}±{e:.4f} -> "
                      f"[{lo:.4f},{hi:.4f}]  {' & '.join(bad)}")
    print(f"      points checked={npts}  data+err y-extent=[{worst_lo:.6g}, {worst_hi:.6g}]")
    if worst_lo >= ylo_lim and worst_hi <= yhi_lim:
        print(f"      OK: ALL error bars inside axis "
              f"(headroom low={worst_lo - ylo_lim:.5f}, high={yhi_lim - worst_hi:.5f})")
    else:
        print("      *** STILL CLIPPED ***")
    return ylo_lim, yhi_lim


for entry in captured:
    print("\n" + "=" * 78)
    print(f"{entry['file']}   size={entry['size_in']} in  dpi={entry['dpi']}")
    print("=" * 78)
    if entry["file"] == "ablation_errorbar.png":
        check_ylim("left  Recall vs rho", entry["axes"][0], df_s, "rho")
        check_ylim("right Recall vs sigma", entry["axes"][1], df_r, "sigma_est")
    else:
        # Fig.1 has both xerr and yerr -> check both directions
        ax = entry["axes"][0]
        x0, x1 = ax["xlim"]
        y0, y1 = ax["ylim"]
        print(f"\n  [trade-off]  xlim=({x0:.6g}, {x1:.6g})   ylim=({y0:.6g}, {y1:.6g})")
        wl, wh, xl, xh = np.inf, -np.inf, np.inf, -np.inf
        for p in SCHEMES:
            for xb, xe, y, ye in zip(df_p[f"{p}_bit_mean"], df_p[f"{p}_bit_std"],
                                     df_p[f"{p}_recall_mean"], df_p[f"{p}_recall_std"]):
                xl = min(xl, xb - xe); xh = max(xh, xb + xe)
                wl = min(wl, y - ye); wh = max(wh, y + ye)
                bad = []
                if xb - xe < x0: bad.append(f"X-LOW by {x0 - (xb - xe):.4g}")
                if xb + xe > x1: bad.append(f"X-HIGH by {(xb + xe) - x1:.4g}")
                if y - ye < y0: bad.append(f"Y-LOW by {y0 - (y - ye):.5f}")
                if y + ye > y1: bad.append(f"Y-HIGH by {(y + ye) - y1:.5f}")
                if bad:
                    print(f"      CLIPPED {p:<10} ({xb:.1f}±{xe:.1f}, {y:.4f}±{ye:.4f}) "
                          f"-> {' & '.join(bad)}")
        print(f"      data+err x-extent=[{xl:.6g}, {xh:.6g}]  y-extent=[{wl:.6g}, {wh:.6g}]")
        ok = (xl >= x0 and xh <= x1 and wl >= y0 and wh <= y1)
        print("      OK: all x/y error bars inside axis" if ok else "      *** STILL CLIPPED ***")
