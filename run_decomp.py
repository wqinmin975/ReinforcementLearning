# -*- coding: utf-8 -*-
"""
审稿人质疑的回应实验：把 EqualGran 与学习式方法之间的差距分解为
「航迹 (trajectory)」与「语义粒度 (granularity)」两个因子。

设计：2x2 因子实验，论文参考点 rho=0.85, sigma_est=1.2，seeds 0-7，
每个 (arm, seed) 用 50 个 episode 评估（与主实验 EVAL_EPISODES 一致）。

  因子 A — 粒度：
    a1 EqualGran : 所有优先级组统一 g=2（EqualGran 规则）
    a2 Fixed     : 固定策略粒度 [g_high, g_mid, g_low] = [2, 1, 1]
  因子 B — 航迹：
    b1 CenterSeek: 朝场景中心（原点）移动的确定性启发式（基线原样）
    b2 Stationary: 每步 dx_code = dy_code = 1（位移 0），UAV 停留在 reset 起点

两个「已发表」的臂 (a1,b1)=EqualGranBaseline 与 (a2,b2)=FixedPolicyBaseline 直接调用
权威文件中的原始类；两个新臂用结构完全相同的 FactorialArm 生成（动作组成方式一致），
并在运行时与原始类做逐位一致性校验。

本文件不修改 uav_semantic_multi_seed_ablation.py。
"""
import os
import numpy as np
import pandas as pd

import uav_semantic_multi_seed_ablation as M

RHO = 0.85
SIGMA = 1.2
N_SEED = 8
EVAL_EPISODES = 50
SAVE_DIR = "./exp_results"


class FactorialArm:
    """通用 2x2 臂：粒度 gran=(g_high, g_mid, g_low)，航迹 mode in {'center','stationary'}。

    评估循环与权威文件 EqualGranBaseline.evaluate / FixedPolicyBaseline.evaluate
    完全同构（同一指标定义：Recall@high 取 episode 最后一步，bits/step 取最后一步
    total_bits；同时记录 episode 内逐步均值 recall_ep / bit_ep 备查）。
    """

    def __init__(self, env, gran, mode):
        self.env = env
        self.gran = tuple(gran)
        assert mode in ("center", "stationary")
        self.mode = mode

    def _move(self):
        if self.mode == "stationary":
            return 1, 1  # delta_inv[1] = 0 -> 不移动
        xu, yu = self.env.xu, self.env.yu
        dx_code = 1 if abs(xu) <= 1.0 else (0 if xu > 0 else 2)
        dy_code = 1 if abs(yu) <= 1.0 else (0 if yu > 0 else 2)
        return dx_code, dy_code

    def evaluate(self, eval_ep):
        recalls, sem_utils, bits = [], [], []
        rec_ep, sem_ep, bit_ep, len_ep = [], [], [], []
        d0_list, df_list = [], []
        for _ in range(eval_ep):
            s, _ = self.env.reset()
            d0_list.append(float(np.hypot(self.env.xu, self.env.yu)))
            done = False
            step_rec, step_sem, step_bit, step_n = 0.0, 0.0, 0.0, 0
            while not done:
                dx_code, dy_code = self._move()
                a = np.array([self.gran[0], self.gran[1], self.gran[2], dx_code, dy_code])
                s_next, _, term, trunc, info = self.env.step(a)
                step_rec += info["recall_high"]
                step_sem += info["sem_utility"]
                step_bit += info["total_bits"]
                step_n += 1
                done = term or trunc
                s = s_next
            recalls.append(info["recall_high"])
            sem_utils.append(info["sem_utility"])
            bits.append(info["total_bits"])
            rec_ep.append(step_rec / step_n)
            sem_ep.append(step_sem / step_n)
            bit_ep.append(step_bit)
            len_ep.append(step_n)
            df_list.append(float(np.hypot(self.env.xu, self.env.yu)))
        return (float(np.mean(recalls)), float(np.mean(sem_utils)), float(np.mean(bits)),
                float(np.mean(rec_ep)), float(np.mean(sem_ep)), float(np.mean(bit_ep)),
                float(np.mean(len_ep)), float(np.mean(d0_list)), float(np.mean(df_list)))


def make_env():
    env = M.UavDynamicSemanticCorrFadingEnv(rho=RHO, sigma_est=SIGMA)
    return env


def run():
    os.makedirs(SAVE_DIR, exist_ok=True)
    arms = [
        ("EqualGran", "CenterSeek", "new"),
        ("EqualGran", "Stationary", "new"),
        ("Fixed", "CenterSeek", "new"),
        ("Fixed", "Stationary", "new"),
    ]
    rows = []
    for seed in range(N_SEED):
        for gran_name, traj_name, _ in arms:
            gran = (2, 2, 2) if gran_name == "EqualGran" else (2, 1, 1)
            mode = "center" if traj_name == "CenterSeek" else "stationary"
            env = make_env()
            env.reset(seed=seed)
            arm = FactorialArm(env, gran, mode)
            rec, sem, bit, rec_ep, sem_ep, bit_ep, elen, d0, dfin = arm.evaluate(EVAL_EPISODES)

            # 一致性校验：新臂 vs 权威文件中已发表的对应类（同 seed、同 env 流）
            env2 = make_env()
            env2.reset(seed=seed)
            if gran_name == "EqualGran" and traj_name == "CenterSeek":
                r2, s2, b2, re2, se2, be2 = M.EqualGranBaseline(env2).evaluate(EVAL_EPISODES, global_g=2)
                shipped = "EqualGranBaseline"
            elif gran_name == "Fixed" and traj_name == "Stationary":
                r2, s2, b2, re2, se2, be2 = M.FixedPolicyBaseline(env2).evaluate(EVAL_EPISODES)
                shipped = "FixedPolicyBaseline"
            else:
                r2 = s2 = b2 = re2 = se2 = be2 = None
                shipped = ""
            if r2 is not None:
                assert abs(r2 - rec) < 1e-12 and abs(b2 - bit) < 1e-12, (
                    f"mismatch vs {shipped}: {r2} vs {rec}, {b2} vs {bit}")
                print(f"  [check] {gran_name}/{traj_name} seed{seed} reproduces {shipped} exactly")

            rows.append(dict(seed=seed, rho=RHO, sigma_est=SIGMA,
                             granularity=gran_name, trajectory=traj_name,
                             gran=str(gran), recall=rec, sem_utility=sem, bits=bit,
                             recall_ep=rec_ep, bit_ep=bit_ep, ep_len=elen,
                             dist_start=d0, dist_final=dfin))
            print(f"seed {seed} | {gran_name:9s} | {traj_name:11s} | "
                  f"recall {rec:.4f} | bits {bit:.1f} | ep_len {elen:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_DIR, "decomp_traj_gran_perseed.csv"), index=False)

    g = df.groupby(["granularity", "trajectory"]).agg(
        recall_mean=("recall", "mean"), recall_std=("recall", lambda x: float(np.std(x, ddof=1))),
        bit_mean=("bits", "mean"), bit_std=("bits", lambda x: float(np.std(x, ddof=1))),
        recall_ep_mean=("recall_ep", "mean"),
        ep_len_mean=("ep_len", "mean"),
        dist_start_mean=("dist_start", "mean"),
        dist_final_mean=("dist_final", "mean"), n=("recall", "size")).reset_index()
    g.to_csv(os.path.join(SAVE_DIR, "decomp_traj_gran.csv"), index=False)
    print("\n=== 2x2 grid means (mean +- sample SD over 8 seeds) ===")
    print(g.to_string(index=False))

    def cell(a, b):
        r = g[(g.granularity == a) & (g.trajectory == b)].iloc[0]
        return r.recall_mean, r.bit_mean

    print("\n=== attribution ===")
    for a in ["EqualGran", "Fixed"]:
        rc, bc = cell(a, "CenterSeek")
        rs, bs = cell(a, "Stationary")
        print(f"[{a}] trajectory effect (CenterSeek - Stationary): recall {rc - rs:+.4f} "
              f"({rc:.4f} vs {rs:.4f}) | bits {bc - bs:+.1f}")
    for b in ["CenterSeek", "Stationary"]:
        re_, be = cell("EqualGran", b)
        rf, bf = cell("Fixed", b)
        print(f"[{b}] granularity effect (EqualGran - Fixed): recall {re_ - rf:+.4f} "
              f"({re_:.4f} vs {rf:.4f}) | bits {be - bf:+.1f}")
    return df, g


if __name__ == "__main__":
    run()
