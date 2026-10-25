# -*- coding: utf-8 -*-
"""Regenerate ablation_errorbar.png and bit_recall_tradeoff.png from the
existing ablation_multi_seed_v2.csv (the dataset the paper's tables use),
using the (corrected) legend labels."""
import os
import sys

# Make the experiment module importable.
sys.path.insert(0, r"D:/ace/语义UAV")

import pandas as pd
import uav_semantic_multi_seed_ablation as m

# Point the module at the results directory so figures are written there.
m.SAVE_DIR = r"D:/ace/语义UAV/exp_results"

csv_path = r"D:/ace/语义UAV/exp_results/ablation_multi_seed_v2.csv"
df = pd.read_csv(csv_path)

m.plot_all_figures(df)

print("Regenerated figures in", m.SAVE_DIR)
for name in ("ablation_errorbar.png", "bit_recall_tradeoff.png"):
    p = os.path.join(m.SAVE_DIR, name)
    print(p, os.path.getmtime(p) if os.path.exists(p) else "MISSING")
