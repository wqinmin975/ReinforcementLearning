# -*- coding: utf-8 -*-
"""
Generate Fig. 3 for the dual-journal submission:
Grid-mean Recall@high under the UNCORRECTED reward across base(g) scaling
factors {0.80, 0.90, 1.00, 1.20} (reference cell rho=0.85, sigma_est=1.2,
4 seeds each). Bars carry +/- one SD. Horizontal reference lines mark the
corrected-reward agents (PPO 0.241, Double-DQN 0.289) and the equal-granularity
heuristic (0.363) to show the uncorrected objective collapses regardless of
coefficient scaling.

Data source: p1_empirical_uncorrected_recall.csv (authoritative empirical run).
"""
import csv
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "p1_empirical_uncorrected_recall.csv")
OUT = os.path.join(HERE, "exp_results", "p1_coeff_empirical.png")

# journal-style serif
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "mathtext.fontset": "stix",
    }
)

scales, means, stds = [], [], []
with open(CSV, newline="") as f:
    for row in csv.DictReader(f):
        scales.append(float(row["base_g_scale"]))
        means.append(float(row["recall_mean"]))
        stds.append(float(row["recall_std"]))

# Reference lines: corrected reward agents + equal-granularity heuristic
REF = [
    (0.363, "EqualGran heuristic (0.363)", "#444444"),
    (0.289, "corrected Double-DQN (0.289)", "#1f77b4"),
    (0.241, "corrected PPO (0.241)", "#d62728"),
]

fig, ax = plt.subplots(figsize=(3.5, 2.9), dpi=300)

x = list(range(len(scales)))
labels = ["0.80", "0.90", "1.00", "1.20"]
bars = ax.bar(
    x,
    means,
    yerr=stds,
    capsize=3.0,
    color="#c9d6e5",
    edgecolor="#2c5d8a",
    linewidth=0.9,
    error_kw={"ecolor": "#2c5d8a", "elinewidth": 0.9},
    width=0.62,
    zorder=3,
)

# value labels above each bar
for xi, m, s in zip(x, means, stds):
    ax.text(
        xi,
        m + s + 0.006,
        "%.3f" % m,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#1a1a1a",
    )

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel(r"base($g$) scaling factor")
ax.set_ylabel(r"Grid-mean Recall@high")
ax.set_ylim(0, 0.41)
ax.set_xlim(-0.7, len(x) - 0.3)

# reference lines
for val, txt, col in REF:
    ax.axhline(val, color=col, linestyle="--", linewidth=0.9, zorder=2)
    ax.text(
        len(x) - 0.35,
        val + 0.004,
        txt,
        ha="right",
        va="bottom",
        fontsize=6.6,
        color=col,
    )

# collapse annotation in the empty mid-band (below the corrected-PPO line)
ax.text(
    -0.55,
    0.13,
    "uncorrected reward:\ncollapse to ~0 at\nevery scaling factor",
    ha="left",
    va="bottom",
    fontsize=6.8,
    color="#7a1f1f",
    style="italic",
)
ax.annotate(
    "",
    xy=(-0.55, 0.012),
    xytext=(-0.55, 0.115),
    arrowprops=dict(arrowstyle="->", color="#7a1f1f", linewidth=0.7),
)

ax.set_title(
    "Recall@high vs base(g) scaling\n(uncorrected reward, reference cell)",
    fontsize=8.5,
)
ax.grid(axis="y", linestyle=":", linewidth=0.5, color="#bbbbbb", zorder=0)

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT, os.path.getsize(OUT), "bytes")
print("data:", list(zip(labels, means, stds)))
