# -*- coding: utf-8 -*-
"""Smoke test for revised reward (A+B+C): 1 config, 1 seed, short run."""
import time
import uav_semantic_multi_seed_ablation as M

# Override scale to minimal for smoke test
M.TRAIN_EPISODES = 60
M.EVAL_EPISODES = 20
M.N_SEED = 1
M.rho_list = [0.85]
M.sigma_est_list = [1.2]
M.SELECTED_CONFIGS = {(0.85, 1.2)}
M.ENABLE_CHECKPOINT = False
M.SAVE_DIR = "exp_results/_smoke"

t0 = time.time()
df, dfc = M.run_ablation_multi_seed()
print("\n=== SMOKE RESULT ===")
print(df.to_string())
print(f"\n[smoke] done in {(time.time()-t0)/60:.2f} min")
# Sanity: PPO/Fixed/EqualGran recall
for col in ["PPO_recall", "DQN_recall", "Fixed_recall", "EqualGran_recall"]:
    if col in df.columns:
        print(f"{col}: {df[col].values}")
