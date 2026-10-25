# -*- coding: utf-8 -*-
"""诊断：PPO checkpoint 实际策略 + 奖励结构分析"""
import sys, numpy as np, torch
sys.path.insert(0, "D:/ace/语义UAV")
import importlib.util
spec = importlib.util.spec_from_file_location("msa", "D:/ace/语义UAV/uav_semantic_multi_seed_ablation.py")
msa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msa)

# 1) 奖励结构静态分析：高优先级区域从 g=1 -> g=2 的边际收益
print("=== 边际收益分析（高优先级区域, s=0.875, SNR=10dB, bw=0.6）===")
snr = 10.0
sig = 1.0 / (1.0 + np.exp(-(snr - 5) / 4))
for g in [1, 2, 3]:
    base = {1: 0.4, 2: 0.7, 3: 0.95}[g]
    cost = {1: 8, 2: 24, 3: 64}[g]
    u = base * sig
    net = 0.875 * u - 0.015 * cost
    print(f"g={g}: U={u:.3f}, weighted={0.875*u:.3f}, bit_pen={0.015*cost:.3f}, net_per_region={net:.3f}")
print()

# 2) 加载 PPO checkpoint 看实际动作分布
env = msa.UavDynamicSemanticCorrFadingEnv(rho=0.85, sigma_est=1.2)
ppo = msa.PPO(env, seed=0)
try:
    ep = ppo.load_checkpoint("D:/ace/语义UAV/exp_results/checkpoints/ppo_rho0.85_sig1.2_seed0.pt")
    print(f"=== PPO checkpoint (rho=0.85, sig=1.2, seed0), ep={ep} ===")
    acts = []
    for _ in range(5):
        s, _ = env.reset()
        done = False
        while not done:
            st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            a = np.array([int(torch.argmax(h(ppo.ac.backbone(st)), dim=-1).item()) for h in ppo.ac.heads])
            acts.append(a.copy())
            s, _, term, trunc, info = env.step(a)
            done = term or trunc
    acts = np.array(acts)
    names = ["g_high", "g_mid", "g_low", "dx", "dy"]
    for i, n in enumerate(names):
        vals, cnts = np.unique(acts[:, i], return_counts=True)
        print(f"{n}: {dict(zip(vals.tolist(), cnts.tolist()))}")
except FileNotFoundError as e:
    print("checkpoint not found:", e)

# 3) 环境层面的 sanity：手动固定动作 [2,2,2,1,1] 的 recall
print()
print("=== Sanity: 固定动作 [2,2,2,1,1] 的 recall_high ===")
env2 = msa.UavDynamicSemanticCorrFadingEnv(rho=0.85, sigma_est=1.2)
s, _ = env2.reset()
done = False
while not done:
    s, _, term, trunc, info = env2.step(np.array([2, 2, 2, 1, 1]))
    done = term or trunc
print("recall_high =", info["recall_high"], "| sem_utility =", info["sem_utility"], "| bits =", info["total_bits"])
