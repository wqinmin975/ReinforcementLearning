# -*- coding: utf-8 -*-
"""
实测 PPO / Double-DQN 的 episode 长度（步数）。

背景：表 X 只有 episode-total bits 与（末步）bits/step，二者相除得到的"步数"是**推算值**，
且因 Jensen 不等式而系统性偏低（T 与每步比特成反比，E[BT]/E[B] <= E[T]）。
Fixed 策略实测 23.60 步 vs 推算 22.97 步，即低估约 2.7%，可证该偏差真实存在。
学习式智能体的每步比特还会随动作变化，推算更不可靠，故直接实测。

做法：加载 v2 八种子训练得到的 *_full.pt checkpoint，用与主实验完全相同的 greedy
评估协议（环境 reset(seed) -> agent -> agent.env.reset(seed) -> 50 episode），
逐步计数直到 terminated or truncated。

不修改 uav_semantic_multi_seed_ablation.py。
"""
import os
import time
import numpy as np
import pandas as pd
import torch

import uav_semantic_multi_seed_ablation as M

SAVE_DIR = "./exp_results"
CK_DIR = os.path.join(SAVE_DIR, "checkpoints")
EVAL_EPISODES = 50


def greedy_action_ppo(agent, s):
    # no_grad 只影响是否构建计算图，不改变 argmax 结果；已逐位校验与 CSV 一致。
    with torch.no_grad():
        st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        head_logits = [h(agent.ac.backbone(st)) for h in agent.ac.heads]
        a = np.empty(len(head_logits), dtype=int)
        for j, hl in enumerate(head_logits):
            if j == 0:
                hl = hl.masked_fill(torch.arange(hl.shape[-1]) < M.HIGH_PRIO_MIN_GRAN, -1e9)
            a[j] = int(torch.argmax(hl, dim=-1).item())
    return a


def greedy_action_dqn(agent, s):
    with torch.no_grad():
        st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        q = agent.q_net(st)
        q = q.masked_fill(~agent.valid_mask, -1e9)
        flat_idx = torch.argmax(q).item()
    return agent._unflatten_action(flat_idx)


def measure(agent, env, kind, eval_ep=EVAL_EPISODES):
    lens = []
    for _ in range(eval_ep):
        s, _ = env.reset()
        done, n = False, 0
        while not done:
            a = greedy_action_ppo(agent, s) if kind == "PPO" else greedy_action_dqn(agent, s)
            s_next, _, term, trunc, _ = env.step(a)
            n += 1
            done = term or trunc
            s = s_next
        lens.append(n)
    return float(np.mean(lens))


def main():
    rows = []
    t_all = time.time()
    for rho in M.rho_list:
        for sigma in M.sigma_est_list:
            for seed in range(8):
                for kind, cls in (("PPO", M.PPO), ("DQN", M.DQN)):
                    ck = os.path.join(CK_DIR, f"{kind.lower()}_rho{rho}_sig{sigma}_seed{seed}_full.pt")
                    if not os.path.exists(ck):
                        print(f"[missing] {ck}")
                        continue
                    env = M.UavDynamicSemanticCorrFadingEnv(rho=rho, sigma_est=sigma)
                    env.reset(seed=seed)
                    agent = cls(env, seed=seed)
                    agent.load_checkpoint(ck)
                    agent.env.reset(seed=seed)
                    t0 = time.time()
                    L = measure(agent, agent.env, kind)
                    rows.append(dict(rho=rho, sigma_est=sigma, seed=seed, method=kind,
                                     ep_len_measured=L))
                    print(f"rho={rho} sig={sigma} seed={seed} {kind}: {L:.3f} steps "
                          f"({time.time()-t0:.1f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_DIR, "ep_len_measured.csv"), index=False)

    g = df.groupby("method").agg(ep_len_mean=("ep_len_measured", "mean"),
                                 ep_len_sd=("ep_len_measured", lambda x: float(np.std(x, ddof=1))),
                                 n=("ep_len_measured", "size")).reset_index()
    print("\n=== measured mean episode length (16 conditions x 8 seeds) ===")
    print(g.round(3).to_string(index=False))

    ps = pd.read_csv(os.path.join(SAVE_DIR, "ablation_perseed_v2.csv"))
    imp = ps.assign(steps=ps.bit_ep / ps.bit).groupby("method")["steps"].mean()
    print("\n=== measured vs implied (bit_ep / bit) ===")
    for _, r in g.iterrows():
        m = {"PPO": "PPO", "DQN": "DQN"}.get(r.method)
        if m in imp.index:
            print(f"  {r.method}: measured {r.ep_len_mean:.3f} | implied {imp[m]:.3f} | "
                  f"bias {r.ep_len_mean - imp[m]:+.3f} ({(r.ep_len_mean/imp[m]-1)*100:+.1f}%)")
    g.to_csv(os.path.join(SAVE_DIR, "ep_len_measured_grid.csv"), index=False)
    print(f"\ntotal wall time {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
