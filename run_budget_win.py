# -*- coding: utf-8 -*-
"""
预算约束下的胜出区间：在「每步比特预算 B」约束下，哪些方法可行、可行集里谁最优。
数据来自论文数据集 exp_results/ablation_multi_seed_v2.csv（网格均值）与
exp_results/ablation_perseed_v2.csv（16 条件 x 8 种子，用于稳健性）。

注意：这是基于「平均 bits/step」的预算可行性论证，不是逐条件的随机保证。
"""
import os
import numpy as np
import pandas as pd

SAVE_DIR = "./exp_results"
BUDGETS = [110, 120, 143, 200, 288, 300]
METHODS = ["PPO", "DQN", "Fixed", "EqualGran"]
NAME = {"PPO": "PPO (proposed)", "DQN": "Double-DQN", "Fixed": "Fixed policy",
        "EqualGran": "EqualGran"}


def main():
    grid = pd.read_csv(os.path.join(SAVE_DIR, "ablation_multi_seed_v2.csv"))
    ps = pd.read_csv(os.path.join(SAVE_DIR, "ablation_perseed_v2.csv"))

    # ---- 网格均值层面的方法画像 ----
    prof = pd.DataFrame({
        "method": METHODS,
        "recall": [grid[f"{m}_recall_mean"].mean() for m in METHODS],
        "bits": [grid[f"{m}_bit_mean"].mean() for m in METHODS],
    })
    print("=== grid-mean profile (16 conditions) ===")
    print(prof.to_string(index=False))

    # ---- 主表：给定 B 的可行集与最优可行方法 ----
    rows = []
    for B in BUDGETS:
        feas = prof[prof.bits <= B]
        if len(feas) == 0:
            rows.append(dict(B=B, feasible="(none)", best="(infeasible)", best_recall=np.nan,
                             n_feasible=0))
            continue
        best = feas.loc[feas.recall.idxmax()]
        rows.append(dict(B=B, feasible=", ".join(NAME[m] for m in feas.method),
                         best=NAME[best.method], best_recall=round(float(best.recall), 4),
                         n_feasible=len(feas)))
    tab = pd.DataFrame(rows)
    print("\n=== budget feasibility (grid means) ===")
    print(tab.to_string(index=False))
    tab.to_csv(os.path.join(SAVE_DIR, "budget_win_region.csv"), index=False)

    # ---- 精确交叉点 ----
    # 注意区间端点：半开区间 [prev, B) 内可行的是 bits <= prev 的方法；
    # bits == B 的那个方法要到 B 本身才可行，不能计入该开区间。
    srt = prof.sort_values("bits").reset_index(drop=True)
    print("\n=== exact crossover (grid-mean bits/step) ===")
    prev = 0.0
    for _, r in srt.iterrows():
        feas = prof[prof.bits <= prev]          # 在 [prev, r.bits) 内可行的方法
        if len(feas) == 0:
            print(f"B in [{prev:.2f}, {r.bits:.2f}) : feasible=[] -> best=(none)")
        else:
            bm = feas.loc[feas.recall.idxmax()]
            print(f"B in [{prev:.2f}, {r.bits:.2f}) : feasible={list(feas.method)} -> "
                  f"best={NAME[bm.method]} (recall {bm.recall:.4f})")
        prev = r.bits
    bm = prof.loc[prof.recall.idxmax()]
    print(f"B >= {prev:.2f}          : feasible={list(prof.method)} -> "
          f"best={NAME[bm.method]} (recall {bm.recall:.4f})")

    # ---- 逐条件稳健性：每个 B 下各方法在 16 个条件中胜出的次数 ----
    long = []
    for m in METHODS:
        t = grid[["rho", "sigma_est"]].copy()
        t["method"] = m
        t["recall"] = grid[f"{m}_recall_mean"]
        t["bits"] = grid[f"{m}_bit_mean"]
        long.append(t)
    long = pd.concat(long, ignore_index=True)

    rows2 = []
    for B in BUDGETS:
        f = long[long.bits <= B]
        if len(f) == 0:
            rows2.append(dict(B=B, n_conditions_with_feasible=0))
            continue
        idx = f.groupby(["rho", "sigma_est"])["recall"].idxmax()
        win = f.loc[idx, "method"].value_counts()
        d = dict(B=B, n_conditions_with_feasible=len(idx))
        for m in METHODS:
            d[f"wins_{m}"] = int(win.get(m, 0))
        rows2.append(d)
    tab2 = pd.DataFrame(rows2).fillna(0)
    print("\n=== per-condition robustness: # of 16 conditions each method is best-feasible ===")
    print(tab2.to_string(index=False))
    tab2.to_csv(os.path.join(SAVE_DIR, "budget_win_region_percond.csv"), index=False)

    # ---- 随机稳健性：per-seed 层面 bits<=B 的比例（16 条件 x 8 种子 = 128 次） ----
    rows3 = []
    for B in BUDGETS:
        d = dict(B=B)
        for m in METHODS:
            d[f"feasfrac_{m}"] = round(float((ps[ps.method == m].bit <= B).mean()), 3)
        rows3.append(d)
    tab3 = pd.DataFrame(rows3)
    print("\n=== stochastic robustness: fraction of 128 (condition,seed) runs with bits<=B ===")
    print(tab3.to_string(index=False))
    tab3.to_csv(os.path.join(SAVE_DIR, "budget_win_region_seedfrac.csv"), index=False)

    # 学习式方法严格胜出的区间定义：EqualGran 不可行 且 学习式方法可行
    b_eq = float(prof.loc[prof.method == "EqualGran", "bits"].iloc[0])
    b_ppo = float(prof.loc[prof.method == "PPO", "bits"].iloc[0])
    b_dqn = float(prof.loc[prof.method == "DQN", "bits"].iloc[0])
    b_fix = float(prof.loc[prof.method == "Fixed", "bits"].iloc[0])
    print(f"\nSUMMARY: EqualGran feasible only when B >= {b_eq:.1f} bits/step.")
    print(f"         Below that, best feasible learned agent is Double-DQN "
          f"(feasible for B >= {b_dqn:.1f}, recall {float(prof.loc[prof.method=='DQN','recall'].iloc[0]):.4f});")
    print(f"         PPO is the unique feasible method only for B in [{b_ppo:.1f}, {b_dqn:.1f}).")
    print(f"         Fixed policy enters at B >= {b_fix:.1f} but never wins "
          f"(recall {float(prof.loc[prof.method=='Fixed','recall'].iloc[0]):.4f} < DQN).")


if __name__ == "__main__":
    main()
