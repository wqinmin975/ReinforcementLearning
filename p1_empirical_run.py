# -*- coding: utf-8 -*-
"""
P1 supplementary ablation — EMPIRICAL confirmation (uncorrected reward).

Goal: show that even under PERTURBED semantic coefficients, the uncorrected
reward (lambda1=0.015, lambda_R=0, no high-prio mask) still drives the PPO
agent to NOT upgrade high-priority regions -> Recall@high collapses near zero.
This corroborates the analytic Root-cause-I result that the misalignment is
structural, not a one-coefficient coincidence.

Does NOT modify the authoritative script: imports it and monkeypatches
module-level globals + semantic_utility. New file only.

Reduced budget: 4 seeds x 800 episodes x 4 base(g)-scaling sets.
Reference cell: rho=0.85, sigma_est=1.2.
"""
import os
import time
import csv
import numpy as np

import uav_semantic_multi_seed_ablation as A

# ---- uncorrected reward (Root-cause-I baseline) ----
A.lambda1 = 0.015
A.lambda_R = 0.0
A.RECALL_SCOPE = "high"
A.HIGH_PRIO_MIN_GRAN = 0

N_SEED = 4
TRAIN_EPISODES = 800
EVAL_EPISODES = 50
RHO, SIGMA = 0.85, 1.2

# base(g) scaling perturbation sets (ratios preserved)
SCALE_SETS = [0.80, 0.90, 1.00, 1.20]

BASE_G_NOM = {1: 0.40, 2: 0.70, 3: 0.95}

def make_sem_util(sf):
    bg = {g: BASE_G_NOM[g] * sf for g in BASE_G_NOM}
    def _f(g, snr_dB):
        if g == 0:
            return 0.0
        return bg[g] * (1.0 / (1.0 + np.exp(-(snr_dB - 5.0) / 4.0)))
    return _f


def main():
    out_rows = []
    t_start = time.time()
    for sf in SCALE_SETS:
        A.semantic_utility = make_sem_util(sf)
        recalls = []
        for seed in range(N_SEED):
            env = A.UavDynamicSemanticCorrFadingEnv(rho=RHO, sigma_est=SIGMA)
            env.reset(seed=seed)
            agent = A.PPO(env, seed=seed)
            agent.train(TRAIN_EPISODES)
            env.reset(seed=seed)
            rec, _, bit, _, _, _ = agent.evaluate(EVAL_EPISODES)
            recalls.append(rec)
            print(f"[sf={sf}] seed{seed} Recall@high(final)={rec:.4f}, bits={bit:.1f}")
        m = float(np.mean(recalls))
        s = float(np.std(recalls, ddof=1)) if len(recalls) > 1 else 0.0
        out_rows.append({"base_g_scale": sf, "recall_mean": m, "recall_std": s})
        print(f">>> sf={sf}: grid-mean Recall@high = {m:.4f} +/- {s:.4f}")
    print(f"\nTotal empirical runtime: {(time.time()-t_start)/60:.1f} min")
    # save
    with open("p1_empirical_uncorrected_recall.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["base_g_scale", "recall_mean", "recall_std"])
        w.writeheader()
        w.writerows(out_rows)
    print("Saved p1_empirical_uncorrected_recall.csv")


if __name__ == "__main__":
    main()
