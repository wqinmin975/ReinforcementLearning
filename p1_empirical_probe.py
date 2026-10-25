# -*- coding: utf-8 -*-
"""
Empirical runtime probe for the P1 supplementary ablation.
Times ONE short PPO training at the reference cell (rho=0.85, sigma_est=1.2)
under the UNCORRECTED reward (lambda1=0.015, lambda_R=0, no high-prio mask)
to estimate the cost of the full empirical run.

Does NOT modify the authoritative script: it imports it and monkeypatches
module-level globals + the semantic_utility function (a fresh replacement,
not an edit to the source file).

Usage: python p1_empirical_probe.py
"""
import time
import numpy as np
import torch

import uav_semantic_multi_seed_ablation as A

# ---- monkeypatch to UNCORRECTED reward (Root-cause-I baseline) ----
A.lambda1 = 0.015          # uncorrected bit-cost weight
A.lambda_R = 0.0           # remove recall-aware correction A for the baseline
A.RECALL_SCOPE = "high"
A.HIGH_PRIO_MIN_GRAN = 0   # no action mask (correction C removed)

# nominal base dict (used by a replacement semantic_utility)
BASE_G = {1: 0.40, 2: 0.70, 3: 0.95}

def patched_semantic_utility(g, snr_dB, _base=BASE_G):
    if g == 0:
        return 0.0
    return _base[g] * (1.0 / (1.0 + np.exp(-(snr_dB - 5.0) / 4.0)))

A.semantic_utility = patched_semantic_utility

# Also patch inside the module namespace used by baselines/DP (they import by name
# at call time from the module, so replacing the attribute is sufficient).


def run_one(n_episodes, seed=0, rho=0.85, sigma_est=1.2):
    env = A.UavDynamicSemanticCorrFadingEnv(rho=rho, sigma_est=sigma_est)
    env.reset(seed=seed)
    agent = A.PPO(env, seed=seed)
    t0 = time.time()
    agent.train(n_episodes)
    dt = time.time() - t0
    env.reset(seed=seed)
    rec, sem, bit, rec_ep, _, _ = agent.evaluate(10)
    return dt, rec, bit


if __name__ == "__main__":
    N = 100  # short probe
    print(f"[probe] short PPO, {N} episodes, 1 seed, uncorrected reward "
          f"(l1={A.lambda1}, lR={A.lambda_R}, mask_min={A.HIGH_PRIO_MIN_GRAN})")
    dt, rec, bit = run_one(N, seed=0)
    print(f"[probe] {N} episodes took {dt:.1f}s  ->  "
          f"~{dt/N:.3f} s/ep.  Eval Recall@high(final)={rec:.3f}, bits={bit:.1f}")
    # estimate full empirical budget: 4 sets x 4 seeds x 800 eps
    per_ep = dt / N
    full_eps = 4 * 4 * 800
    est = per_ep * full_eps
    print(f"[probe] ESTIMATED full empirical budget "
          f"(4 sets x 4 seeds x 800 eps) = {est/60:.1f} min "
          f"({est/3600:.2f} h) on this CPU.")
