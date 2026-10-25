# -*- coding: utf-8 -*-
"""
Task B：逐步（per-step）回合均值口径下的 2x2 稳健性复核。

动机：论文主指标 Recall@high 取 episode 最后一步，而核心机制（能量续航：
288 bits/step 只活 ~9.8 步、144 bits/step 活 ~17.9 步）是**整段回合**的机制。
若只看末步，优势可能被放大。故在逐步均值口径下复核。

四臂（与 run_decomp.py 完全一致）：
    a1 EqualGran  : 粒度 (2,2,2)
    a2 Fixed      : 粒度 (2,1,1)
    b1 CenterSeek : 朝场景中心的确定性启发式
    b2 Stationary : dx_code = dy_code = 1（不移动）

两个窗口：
  * 窗口 1（生存窗口，K=20）：前 20 步。同时报两种处理
        - zero_fill  : 超过 episode 终止时刻的步按 recall = 0 计入（与窗口 2 同规则）
        - alive_only : 只对实际存活的步取均值，最多取 20 步
  * 窗口 2（全回合，K=200）：全部 200 步，终止后其决策冻结、召回按 0 计（zero_fill）。

评估协议与论文主实验一致：rho=0.85, sigma_est=1.2, seeds 0-7, 每 seed 50 个 episode。
不修改 uav_semantic_multi_seed_ablation.py。
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

from run_decomp import FactorialArm, make_env, RHO, SIGMA, N_SEED, EVAL_EPISODES

SAVE_DIR = "./exp_results"
K_SHORT = 20
K_FULL = 200

ARMS = [("EqualGran", "CenterSeek"), ("EqualGran", "Stationary"),
        ("Fixed", "CenterSeek"), ("Fixed", "Stationary")]


def run_arm_episodes(gran_name, traj_name, seed, eval_ep):
    """返回该 (臂, seed) 下每个 episode 的逐步 recall 列表与长度。"""
    gran = (2, 2, 2) if gran_name == "EqualGran" else (2, 1, 1)
    mode = "center" if traj_name == "CenterSeek" else "stationary"
    env = make_env()
    env.reset(seed=seed)
    arm = FactorialArm(env, gran, mode)
    seqs, lens = [], []
    for _ in range(eval_ep):
        env.reset()
        done, step_n = False, 0
        rec_seq = []
        while not done:
            dx, dy = arm._move()
            a = np.array([gran[0], gran[1], gran[2], dx, dy])
            _, _, term, trunc, info = env.step(a)
            rec_seq.append(info["recall_high"])
            step_n += 1
            done = term or trunc
        seqs.append(np.asarray(rec_seq, dtype=float))
        lens.append(step_n)
    return seqs, lens


def window_means(seqs, K, mode):
    """每个 episode 在窗口 K 内的逐步均值。
    mode='zero_fill'  : 终止后的步按 0 计入，分母固定为 K
    mode='alive_only' : 只对存活步取均值，分母为 min(T, K)"""
    out = []
    for s in seqs:
        T = len(s)
        k = min(T, K)
        if k == 0:
            out.append(0.0)
            continue
        if mode == "zero_fill":
            out.append(float(s[:k].sum()) / K)
        else:
            out.append(float(s[:k].sum()) / k)
    return np.asarray(out)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    rows = []
    for seed in range(N_SEED):
        for gran_name, traj_name in ARMS:
            seqs, lens = run_arm_episodes(gran_name, traj_name, seed, EVAL_EPISODES)
            w1z = window_means(seqs, K_SHORT, "zero_fill")
            w1a = window_means(seqs, K_SHORT, "alive_only")
            w2z = window_means(seqs, K_FULL, "zero_fill")
            # 末步口径（论文主指标）作为对照
            final = np.asarray([s[-1] if len(s) else 0.0 for s in seqs])
            rows.append(dict(seed=seed, granularity=gran_name, trajectory=traj_name,
                             w1_20_zero=float(w1z.mean()),
                             w1_20_alive=float(w1a.mean()),
                             w2_200_zero=float(w2z.mean()),
                             final_step=float(final.mean()),
                             ep_len=float(np.mean(lens))))
            print(f"seed {seed} | {gran_name:9s} | {traj_name:11s} | "
                  f"w1_zero {w1z.mean():.4f} | w1_alive {w1a.mean():.4f} | "
                  f"w2_zero {w2z.mean():.4f} | final {final.mean():.4f} | len {np.mean(lens):.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_DIR, "persistep_means.csv"), index=False)

    metrics = ["w1_20_zero", "w1_20_alive", "w2_200_zero", "final_step"]
    g = df.groupby(["granularity", "trajectory"]).agg(
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_sd": (m, lambda x: float(np.std(x, ddof=1))) for m in metrics},
        ep_len=("ep_len", "mean"), n=("seed", "size")).reset_index()
    g.to_csv(os.path.join(SAVE_DIR, "persistep_means_grid.csv"), index=False)
    print("\n=== per-arm means (8 seeds, mean +- sample SD) ===")
    print(g.round(4).to_string(index=False))

    def cell(gran, traj, m):
        return df[(df.granularity == gran) & (df.trajectory == traj)].sort_values("seed")[m].values

    def paired(a, b, metric, label):
        x = cell(a[0], a[1], metric) - cell(b[0], b[1], metric)
        n = len(x); m = x.mean(); s = x.std(ddof=1); se = s / np.sqrt(n)
        t, p = stats.ttest_1samp(x, 0)
        ci = stats.t.interval(0.95, n - 1, loc=m, scale=se)
        print(f"  {label:46s} diff {m:+.4f} SD {s:.4f} 95%CI [{ci[0]:+.4f},{ci[1]:+.4f}] "
              f"t={t:+.3f} p={p:.3e} (n={n})")
        return dict(metric=metric, contrast=label, diff=m, sd=s,
                    ci_lo=ci[0], ci_hi=ci[1], t=t, p=p, n=n,
                    mean_A=cell(a[0], a[1], metric).mean(),
                    mean_B=cell(b[0], b[1], metric).mean())

    print("\n=== pure-flight contrasts (step-mean metrics) ===")
    out = []
    for metric in metrics:
        print(f"\n[{metric}]")
        d1 = paired(("Fixed", "CenterSeek"), ("Fixed", "Stationary"), metric,
                   "Fixed granularity: CenterSeek - Stationary")
        d1.update(axis="flight"); out.append(d1)
        d2 = paired(("EqualGran", "CenterSeek"), ("EqualGran", "Stationary"), metric,
                   "EqualGran: CenterSeek - Stationary")
        d2.update(axis="flight"); out.append(d2)

    print("\n=== pure-granularity contrast (CenterSeek trajectory; granularity axis) ===")
    for metric in metrics:
        print(f"\n[{metric}]")
        d3 = paired(("Fixed", "CenterSeek"), ("EqualGran", "CenterSeek"), metric,
                   "CenterSeek: Fixed(2,1,1) - EqualGran(2,2,2)")
        d3.update(axis="granularity", metric=metric); out.append(d3)

    cdf = pd.DataFrame(out)
    cdf.to_csv(os.path.join(SAVE_DIR, "persistep_means_contrasts.csv"), index=False)
    print("\n=== dominance check: Fixed+CenterSeek vs EqualGran+CenterSeek ===")
    for metric in metrics:
        d = cdf[(cdf.metric == metric) & (cdf.contrast.str.startswith("CenterSeek: Fixed"))].iloc[0]
        bits_ok = True  # bits: 144.7 vs 288.0，与窗口无关
        print(f"  [{metric}] recall diff {d['diff']:+.4f} (p={d['p']:.3e}); "
              f"bits 144.7 < 288.0 -> dominance holds: "
              f"{'YES' if (d['diff'] > 0 and d['p'] < 0.05) else 'NO'}")
    return df, g, cdf


if __name__ == "__main__":
    main()
