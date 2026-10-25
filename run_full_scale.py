# -*- coding: utf-8 -*-
"""
正式全规模运行启动器（后台任务）
================================
- 固定使用 uav_semantic_multi_seed_ablation.py 中的默认全规模参数：
    TRAIN_EPISODES=800, EVAL_EPISODES=50, N_SEED=3,
    rho_list=[0.5,0.7,0.85,0.95] × sigma_est_list=[0.0,0.6,1.2,2.0]  => 4x4=16 组
- ENABLE_CHECKPOINT=True：每 50 回合落盘 + 末态存档，中断后重跑自动续训。
- 把 torch 线程数拉满（12 逻辑核）以加速 CPU 训练。
- 所有 stdout/stderr 同步写入 run_full_scale.log 便于监控。

监控方式：
    python -c "print(open('run_full_scale.log').read())"   # 或 tail
    python -c "import pandas as pd; print(pd.read_csv('exp_results/ablation_multi_seed.csv'))"
"""
import sys
import time
import torch

# 在 import 主模块之前设置线程数，避免任何 torch 算子提前绑定默认线程池
N_THREADS = 12
torch.set_num_threads(N_THREADS)
torch.set_num_interop_threads(N_THREADS)
print(f"[launcher] torch threads set to {N_THREADS}", flush=True)

import uav_semantic_multi_seed_ablation as M

# 显式确认运行规模（若想缩小重测可在此处覆盖模块级常量）
print("=== FULL-SCALE RUN CONFIG ===", flush=True)
print(f"TRAIN_EPISODES = {M.TRAIN_EPISODES}", flush=True)
print(f"EVAL_EPISODES  = {M.EVAL_EPISODES}", flush=True)
print(f"N_SEED         = {M.N_SEED}", flush=True)
print(f"rho_list       = {M.rho_list}", flush=True)
print(f"sigma_est_list = {M.sigma_est_list}", flush=True)
print(f"grid configs   = {len(M.rho_list) * len(M.sigma_est_list)}", flush=True)
print(f"ENABLE_CHECKPOINT = {M.ENABLE_CHECKPOINT}", flush=True)
print(f"SELECTED_CONFIGS = {M.SELECTED_CONFIGS}", flush=True)
print(f"SAVE_DIR       = {M.SAVE_DIR}", flush=True)
print("=============================", flush=True)

t0 = time.time()
df_result, df_complex = M.run_ablation_multi_seed()
M.plot_all_figures(df_result)

elapsed = time.time() - t0
print(f"\n[launcher] DONE in {elapsed/3600:.2f} h", flush=True)
print("\n===== Result (ablation_multi_seed.csv) =====", flush=True)
print(df_result.to_string(), flush=True)
print("\n===== Complexity =====", flush=True)
print(df_complex.to_string(), flush=True)
