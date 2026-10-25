# -*- coding: utf-8 -*-
"""Probe the CSV to quantify Fig.1 overlap and Fig.2 error-bar clipping."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"D:/ace/语义UAV")
import uav_semantic_multi_seed_ablation as m

df = pd.read_csv(r"D:/ace/语义UAV/exp_results/ablation_multi_seed.csv")
rho_list = sorted(df["rho"].unique())
sig_list = sorted(df["sigma_est"].unique())
fix_sigma = m.sigma_est_list[len(m.sigma_est_list) // 2]
fix_rho = m.rho_list[len(m.rho_list) // 2]
print("rho_list   :", rho_list)
print("sigma_list :", sig_list)
print("fix_sigma  :", fix_sigma, " fix_rho:", fix_rho)

METHODS = [("PPO", "PPO (corrected)"), ("DQN", "Double-DQN"),
           ("Fixed", "Fixed policy"), ("EqualGran", "EqualGran")]


def report(name, xvals, xcol_base, ycol_base, yerr=True, xerr=False, xlab=""):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    xs = list(xvals)
    ylo, yhi = [], []
    xlo, xhi = [], []
    print(f"{'method':<14} {'x':>10} {'x-left':>10} {'x-right':>10} "
          f"{'y':>9} {'y-low':>9} {'y-up':>9}")
    for pre, _ in METHODS:
        for xv in xs:
            rowselection = None
            pass
        # handled per-axis below
    return xs, None


# --- Fig.2 left axis: Recall vs rho at fixed sigma ---
df_s = df[np.isclose(df["sigma_est"], fix_sigma)].sort_values("rho")
print("\n" + "=" * 78)
print("FIG.2 LEFT AXIS  Recall vs rho (sigma_est=%s)" % fix_sigma)
print("=" * 78)
print(f"{'method':<12}{'x(rho)':>9}{'y':>10}{'y-std':>10}{'y_low':>10}{'y_up':>10}")
ymin_all, ymax_all = np.inf, -np.inf
for pre, _ in METHODS:
    for x, y, e in zip(df_s["rho"], df_s[f"{pre}_recall_mean"], df_s[f"{pre}_recall_std"]):
        lo, hi = y - e, y + e
        ymin_all = min(ymin_all, lo); ymax_all = max(ymax_all, hi)
        print(f"{pre:<12}{x:>9.2f}{y:>10.4f}{e:>10.4f}{lo:>10.4f}{hi:>10.4f}")
print(f"  --> data+err y-range: [{ymin_all:.4f}, {ymax_all:.4f}]  "
      f"span={ymax_all - ymin_all:.4f}")

# --- Fig.2 right axis: Recall vs sigma at fixed rho ---
df_r = df[np.isclose(df["rho"], fix_rho)].sort_values("sigma_est")
print("\n" + "=" * 78)
print("FIG.2 RIGHT AXIS  Recall vs sigma_est (rho=%s)" % fix_rho)
print("=" * 78)
print(f"{'method':<12}{'x(sig)':>9}{'y':>10}{'y-std':>10}{'y_low':>10}{'y_up':>10}")
y2min, y2max = np.inf, -np.inf
for pre, _ in METHODS:
    for x, y, e in zip(df_r["sigma_est"], df_r[f"{pre}_recall_mean"], df_r[f"{pre}_recall_std"]):
        lo, hi = y - e, y + e
        y2min = min(y2min, lo); y2max = max(y2max, hi)
        print(f"{pre:<12}{x:>9.2f}{y:>10.4f}{e:>10.4f}{lo:>10.4f}{hi:>10.4f}")
print(f"  --> data+err y-range: [{y2min:.4f}, {y2max:.4f}]  "
      f"span={y2max - y2min:.4f}")

# --- Fig.1: bit-recall trade-off ---
df_p = df[np.isclose(df["rho"], fix_rho)]
print("\n" + "=" * 92)
print("FIG.1  Bit-Recall trade-off (rho=%s)" % fix_rho)
print("=" * 92)
print(f"{'method':<12}{'xb':>12}{'xerr':>10}{'x_lo':>12}{'x_hi':>12}"
      f"{'y':>10}{'y-std':>10}{'y_lo':>10}{'y_hi':>10}")
for pre, _ in METHODS:
    for xb, xe, y, ye in zip(df_p[f"{pre}_bit_mean"], df_p[f"{pre}_bit_std"],
                             df_p[f"{pre}_recall_mean"], df_p[f"{pre}_recall_std"]):
        print(f"{pre:<12}{xb:>12.1f}{xe:>10.1f}{xb-xe:>12.1f}{xb+xe:>12.1f}"
              f"{y:>10.4f}{ye:>10.4f}{y-ye:>10.4f}{y+ye:>10.4f}")

# ---- quantify OVERLAP in Fig.1 among the 4 method point clusters ----
print("\n" + "=" * 78)
print("FIG.1 OVERLAP ANALYSIS (per sigma_est point index)")
print("=" * 78)
cent = {}
for pre, _ in METHODS:
    cent[pre] = list(zip(df_p[f"{pre}_bit_mean"].values, df_p[f"{pre}_recall_mean"].values,
                         df_p[f"{pre}_bit_std"].values, df_p[f"{pre}_recall_std"].values))
n = len(cent["PPO"])
# typical marker size in points -> approximate half-width in data units
# will be measured after drawing; here report pairwise index distances in units of xerr
for i in range(n):
    print(f"\n point #{i} (sigma_est={df_p['sigma_est'].values[i]}):")
    for a_i, (pa, _) in enumerate(METHODS):
        for pb, _ in METHODS[(a_i + 1):]:
            xa, ya, exa, _ = cent[pa][i]
            xb_, yb_, exb, _ = cent[pb][i]
            dx = abs(xa - xb_)
            dy = abs(ya - yb_)
            # separation normalised by the mean error-bar half-width
            sep_x = dx / max((exa + exb) / 2, 1e-9)
            print(f"   {pa:<10} vs {pb:<10} dx={dx:>9.1f} ({(exa+exb)/2:>7.1f} err hw)"
                  f" -> {sep_x:>5.2f}x errbar | dy={dy:>8.4f}")
