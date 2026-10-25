# -*- coding: utf-8 -*-
"""
切面：学习式智能体的每步比特开销随"回合内相对位置"的变化，并同时记录该位置的
UAV-中心距离，用于检验假设：临近终点时 UAV 已靠近场景中心、SNR 更好，因而值得花更多比特。

做法：加载 v2 的 *_full.pt checkpoint（16 格 × 8 种子），greedy 评估 50 episode，
逐步记录 (total_bits, 距中心距离)，按"步序号 / 该 episode 总步数"归一化位置分三档
（前 1/3、中 1/3、后 1/3），逐档汇总。

不修改 uav_semantic_multi_seed_ablation.py。
"""
import os
import numpy as np
import pandas as pd
import torch

import uav_semantic_multi_seed_ablation as M
from measure_ep_len import greedy_action_ppo, greedy_action_dqn

SAVE_DIR = "./exp_results"
CK_DIR = os.path.join(SAVE_DIR, "checkpoints")
EVAL_EPISODES = 50


def collect(agent, env, kind, eval_ep=EVAL_EPISODES):
    """返回 [(rel_pos, bits, dist), ...] 的逐步记录。"""
    rec = []
    for _ in range(eval_ep):
        s, _ = env.reset()
        done = False
        bits, dists = [], []
        while not done:
            a = greedy_action_ppo(agent, s) if kind == "PPO" else greedy_action_dqn(agent, s)
            s_next, _, term, trunc, info = env.step(a)
            bits.append(info["total_bits"])
            dists.append(float(np.hypot(env.xu, env.yu)))
            done = term or trunc
            s = s_next
        T = len(bits)
        for k in range(T):
            rec.append(((k + 0.5) / T, bits[k], dists[k]))
    return rec


def main():
    rows = []
    for kind, cls in (("PPO", M.PPO), ("DQN", M.DQN)):
        for rho in M.rho_list:
            for sigma in M.sigma_est_list:
                for seed in range(8):
                    ck = os.path.join(CK_DIR, f"{kind.lower()}_rho{rho}_sig{sigma}_seed{seed}_full.pt")
                    if not os.path.exists(ck):
                        continue
                    env = M.UavDynamicSemanticCorrFadingEnv(rho=rho, sigma_est=sigma)
                    env.reset(seed=seed)
                    agent = cls(env, seed=seed)
                    agent.load_checkpoint(ck)
                    agent.env.reset(seed=seed)
                    for rp, b, dd in collect(agent, agent.env, kind):
                        rows.append(dict(method=kind, rho=rho, sigma_est=sigma, seed=seed,
                                         rel_pos=rp, bits=b, dist=dd))
    df = pd.DataFrame(rows)
    df["third"] = pd.cut(df.rel_pos, [0.0, 1 / 3, 2 / 3, 1.0],
                         labels=["first", "middle", "last"], include_lowest=True)
    df.to_csv(os.path.join(SAVE_DIR, "bit_profile_perseed.csv"), index=False)

    g = (df.groupby(["method", "third"], observed=True)
           .agg(bits=("bits", "mean"), dist=("dist", "mean"), n=("bits", "size"))
           .reset_index())
    print("\n=== per-step bits and distance-to-centre by position within episode ===")
    print(g.round(3).to_string(index=False))
    g.to_csv(os.path.join(SAVE_DIR, "bit_profile_by_third.csv"), index=False)

    print("\n=== last-third vs first-third ===")
    for m in ["PPO", "DQN"]:
        f = g[(g["method"] == m) & (g.third == "first")].iloc[0]
        l = g[(g["method"] == m) & (g.third == "last")].iloc[0]
        print(f"  {m}: bits {f.bits:.2f} -> {l.bits:.2f} ({(l.bits/f.bits-1)*100:+.1f}%) | "
              f"dist {f.dist:.2f} -> {l.dist:.2f} ({(l.dist/f.dist-1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
