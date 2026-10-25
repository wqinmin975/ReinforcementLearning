# -*- coding: utf-8 -*-
"""
评审整改实验驱动（C2 / M3 / M4 / M5 / M7）
==========================================
复用 uav_semantic_multi_seed_ablation 中的环境 / 智能体，但参数化：
  - 召回项作用域 RECALL_SCOPE（'high' = 与 (7) 同构；'all' = 旧实现）
  - 奖励权重 lambda1 / lambda_R
  - 高优先级掩码 HIGH_PRIO_MIN_GRAN（0 = 关闭）
  - PPO 激活函数（tanh / relu，用于 M5 统一激活）

并新增：每回合（episode）级指标（时间平均 recall / 总比特）—— 回应 M7。

运行：
  python run_review_experiments.py smoke      # 冒烟测试（1 配置 1 种子 少量训练）
  python run_review_experiments.py bench      # 计时 1 次 PPO/DQN 训练以估算总时长
  python run_review_experiments.py main       # 主网格重跑（C2+M4）：修正奖励, N_SEED=8, 全 16 网格
  python run_review_experiments.py m3         # 修正机制消融（M3）：4 变体 × 代表配置 × 仅 PPO
  python run_review_experiments.py m5         # 统一激活对比（M5）：PPO tanh vs relu
  python run_review_experiments.py all        # 依次 main + m3 + m5
  python run_review_experiments.py sig        # 基于 main 的 per-seed 结果做显著性检验（M4）
"""
import sys, os, time, json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uav_semantic_multi_seed_ablation as M

REP_CONFIGS = [(0.5, 0.0), (0.5, 1.2), (0.5, 2.0),
               (0.85, 0.0), (0.85, 1.2), (0.85, 2.0),
               (0.95, 0.0), (0.95, 1.2), (0.95, 2.0)]

# 修正后奖励（论文主线）：lambda1 0.008, lambda_R 0.4, 掩码开, scope=high
CFG_FULL = dict(lambda1=0.008, lambda_R=0.4, mask_gran=2, scope="high",
                tag="full", agents={"PPO", "DQN", "Fixed", "EqualGran"})
CFG_NORECALL = dict(lambda1=0.008, lambda_R=0.0, mask_gran=2, scope="high",
                    tag="norecall", agents={"PPO", "Fixed", "EqualGran"})
CFG_NOBIT = dict(lambda1=0.015, lambda_R=0.4, mask_gran=2, scope="high",
                 tag="nobitrew", agents={"PPO", "Fixed", "EqualGran"})
CFG_NOMASK = dict(lambda1=0.008, lambda_R=0.4, mask_gran=0, scope="high",
                  tag="nomask", agents={"PPO", "Fixed", "EqualGran"})


def run_one_config(rho, sigma, seeds, cfg):
    """运行单个 (rho, sigma) 配置下的方案集合，返回 (result_row, perseed_rows)。"""
    M.lambda1 = cfg["lambda1"]
    M.lambda_R = cfg["lambda_R"]
    M.HIGH_PRIO_MIN_GRAN = cfg["mask_gran"]
    M.RECALL_SCOPE = cfg["scope"]
    ppo_act = cfg.get("ppo_activation", "tanh")
    agents = cfg.get("agents", {"PPO", "DQN", "Fixed", "EqualGran"})
    tag = cfg.get("tag", "")
    suf = ("_" + tag) if tag else ""
    ck_dir = os.path.join(M.SAVE_DIR, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)

    p_r, p_s, p_b, p_re, p_be = [], [], [], [], []
    d_r, d_s, d_b, d_re, d_be = [], [], [], [], []
    f_r, f_s, f_b, f_re, f_be = [], [], [], [], []
    e_r, e_s, e_b, e_re, e_be = [], [], [], [], []
    perseed = []

    for seed in seeds:
        if "PPO" in agents:
            env = M.UavDynamicSemanticCorrFadingEnv(rho=rho, sigma_est=sigma)
            env.reset(seed=seed)
            ppo = M.PPO(env, seed=seed, activation=ppo_act)
            ck = os.path.join(ck_dir, f"ppo_rho{rho}_sig{sigma}_seed{seed}{suf}.pt")
            st = 0
            if M.ENABLE_CHECKPOINT and os.path.exists(ck):
                st = ppo.load_checkpoint(ck)
            if st < M.TRAIN_EPISODES:
                ppo.train(M.TRAIN_EPISODES, resume_from=st,
                           ckpt_path=ck if M.ENABLE_CHECKPOINT else None, save_every=50)
            ppo.env.reset(seed=seed)
            a, b, c, d, e, f = ppo.evaluate(M.EVAL_EPISODES)
            p_r.append(a); p_s.append(b); p_b.append(c); p_re.append(d); p_be.append(f)
            perseed.append(dict(rho=rho, sigma_est=sigma, tag=tag, method="PPO", seed=seed,
                                recall=a, sem=b, bit=c, recall_ep=d, bit_ep=f))
        if "DQN" in agents:
            env2 = M.UavDynamicSemanticCorrFadingEnv(rho=rho, sigma_est=sigma)
            env2.reset(seed=seed)
            dqn = M.DQN(env2, seed=seed)
            ck2 = os.path.join(ck_dir, f"dqn_rho{rho}_sig{sigma}_seed{seed}{suf}.pt")
            st2 = 0
            if M.ENABLE_CHECKPOINT and os.path.exists(ck2):
                st2 = dqn.load_checkpoint(ck2)
            if st2 < M.TRAIN_EPISODES:
                dqn.train(M.TRAIN_EPISODES, resume_from=st2,
                           ckpt_path=ck2 if M.ENABLE_CHECKPOINT else None, save_every=50)
            dqn.env.reset(seed=seed)
            a, b, c, d, e, f = dqn.evaluate(M.EVAL_EPISODES)
            d_r.append(a); d_s.append(b); d_b.append(c); d_re.append(d); d_be.append(f)
            perseed.append(dict(rho=rho, sigma_est=sigma, tag=tag, method="DQN", seed=seed,
                                recall=a, sem=b, bit=c, recall_ep=d, bit_ep=f))
        if "Fixed" in agents:
            envf = M.UavDynamicSemanticCorrFadingEnv(rho=rho, sigma_est=sigma)
            envf.reset(seed=seed)
            fbl = M.FixedPolicyBaseline(envf)
            a, b, c, d, e, f = fbl.evaluate(M.EVAL_EPISODES)
            f_r.append(a); f_s.append(b); f_b.append(c); f_re.append(d); f_be.append(f)
            perseed.append(dict(rho=rho, sigma_est=sigma, tag=tag, method="Fixed", seed=seed,
                                recall=a, sem=b, bit=c, recall_ep=d, bit_ep=f))
        if "EqualGran" in agents:
            enve = M.UavDynamicSemanticCorrFadingEnv(rho=rho, sigma_est=sigma)
            enve.reset(seed=seed)
            ebl = M.EqualGranBaseline(enve)
            a, b, c, d, e, f = ebl.evaluate(M.EVAL_EPISODES, global_g=2)
            e_r.append(a); e_s.append(b); e_b.append(c); e_re.append(d); e_be.append(f)
            perseed.append(dict(rho=rho, sigma_est=sigma, tag=tag, method="EqualGran", seed=seed,
                                recall=a, sem=b, bit=c, recall_ep=d, bit_ep=f))

    def col(xs):
        return (np.mean(xs) if xs else None, M._sstd(xs))

    row = {"rho": rho, "sigma_est": sigma, "tag": tag}
    for name, rl, sl, bl, rel, bel in [
        ("PPO", p_r, p_s, p_b, p_re, p_be),
        ("DQN", d_r, d_s, d_b, d_re, d_be),
        ("Fixed", f_r, f_s, f_b, f_re, f_be),
        ("EqualGran", e_r, e_s, e_b, e_re, e_be),
    ]:
        rm, rs = col(rl); sm, ss = col(sl); bm, bs = col(bl)
        rem, res = col(rel); bem, bes = col(bel)
        row[f"{name}_recall_mean"] = rm; row[f"{name}_recall_std"] = rs
        row[f"{name}_sem_mean"] = sm; row[f"{name}_sem_std"] = ss
        row[f"{name}_bit_mean"] = bm; row[f"{name}_bit_std"] = bs
        row[f"{name}_recall_ep_mean"] = rem; row[f"{name}_recall_ep_std"] = res
        row[f"{name}_bit_ep_mean"] = bem; row[f"{name}_bit_ep_std"] = bes
    return row, perseed


def run_grid(cfg, seeds, configs, out_csv, perseed_csv=None):
    rows, perseeds = [], []
    for rho, sigma in configs:
        print(f"\n##### rho={rho}, sigma_est={sigma} seed={seeds} #####")
        t0 = time.time()
        row, ps = run_one_config(rho, sigma, seeds, cfg)
        rows.append(row); perseeds.extend(ps)
        print(f"  PPO recall {row.get('PPO_recall_mean')} | DQN recall {row.get('DQN_recall_mean')} "
              f"| EqGran recall {row.get('EqualGran_recall_mean')} | {time.time()-t0:.1f}s")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    if perseed_csv is not None:
        pd.DataFrame(perseeds).to_csv(perseed_csv, index=False)
    print(f"\nSaved {out_csv}")
    return df


def run_main(seeds):
    M.ENABLE_CHECKPOINT = True
    cfg = dict(CFG_FULL, agents={"PPO", "DQN", "Fixed", "EqualGran"})
    configs = [(r, s) for r in M.rho_list for s in M.sigma_est_list]
    df = run_grid(cfg, seeds, configs,
                  os.path.join(M.SAVE_DIR, "ablation_multi_seed_v2.csv"),
                  os.path.join(M.SAVE_DIR, "ablation_perseed_v2.csv"))
    M.plot_all_figures(df)
    # 复制图到论文 figures 目录
    for fn in ["ablation_errorbar.png", "bit_recall_tradeoff.png"]:
        src = os.path.join(M.SAVE_DIR, fn)
        for dst in ["stage1/figures", "stage2/figures"]:
            try:
                import shutil
                shutil.copy(src, os.path.join(dst, fn))
            except Exception as ex:
                print("  copy fig warn:", ex)
    return df


def run_m3(seeds):
    M.ENABLE_CHECKPOINT = True
    variants = [CFG_FULL, CFG_NORECALL, CFG_NOBIT, CFG_NOMASK]
    all_rows, all_perseed = [], []
    for cfg in variants:
        print(f"\n===== M3 变体 {cfg['tag']} =====")
        rows, perseeds = [], []
        for rho, sigma in REP_CONFIGS:
            row, ps = run_one_config(rho, sigma, seeds, cfg)
            rows.append(row); perseeds.extend(ps)
        all_rows.extend(rows); all_perseed.extend(perseeds)
        pd.DataFrame(rows).to_csv(
            os.path.join(M.SAVE_DIR, f"ablation_m3_{cfg['tag']}.csv"), index=False)
    pd.DataFrame(all_rows).to_csv(os.path.join(M.SAVE_DIR, "ablation_m3_all.csv"), index=False)
    pd.DataFrame(all_perseed).to_csv(os.path.join(M.SAVE_DIR, "ablation_m3_perseed.csv"), index=False)
    print("\nSaved M3 results.")


def run_m5(seeds):
    M.ENABLE_CHECKPOINT = True
    all_rows, all_perseed = [], []
    for act in ["tanh", "relu"]:
        cfg = dict(CFG_FULL, tag=f"m5_{act}", ppo_activation=act,
                   agents={"PPO", "Fixed", "EqualGran"})
        print(f"\n===== M5 PPO activation={act} =====")
        rows, perseeds = [], []
        for rho, sigma in REP_CONFIGS:
            row, ps = run_one_config(rho, sigma, seeds, cfg)
            rows.append(row); perseeds.extend(ps)
        all_rows.extend(rows); all_perseed.extend(perseeds)
        pd.DataFrame(rows).to_csv(
            os.path.join(M.SAVE_DIR, f"ablation_m5_{act}.csv"), index=False)
    pd.DataFrame(all_rows).to_csv(os.path.join(M.SAVE_DIR, "ablation_m5_all.csv"), index=False)
    pd.DataFrame(all_perseed).to_csv(os.path.join(M.SAVE_DIR, "ablation_m5_perseed.csv"), index=False)
    print("\nSaved M5 results.")


def run_significance():
    """M4：基于 main 的 per-seed 结果，对各方案对的 recall 做配对 t 检验。"""
    try:
        from scipy import stats
    except ImportError:
        print("scipy 未安装，跳过显著性检验（pip install scipy）")
        return
    ps = pd.read_csv(os.path.join(M.SAVE_DIR, "ablation_perseed_v2.csv"))
    out = []
    for (rho, sigma), grp in ps.groupby(["rho", "sigma_est"]):
        methods = grp["method"].unique()
        for m1 in methods:
            for m2 in methods:
                if m1 >= m2:
                    continue
                a = grp[grp.method == m1].sort_values("seed")["recall"].values
                b = grp[grp.method == m2].sort_values("seed")["recall"].values
                if len(a) >= 2 and len(b) >= 2 and len(a) == len(b):
                    t, p = stats.ttest_rel(a, b)
                    out.append(dict(rho=rho, sigma_est=sigma, m1=m1, m2=m2,
                                    mean1=np.mean(a), mean2=np.mean(b),
                                    t=t, p=p, n=len(a)))
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(M.SAVE_DIR, "significance_v2.csv"), index=False)
    # 汇总：哪些对比在多数配置上显著（p<0.05）
    sig_pairs = df[df.p < 0.05].groupby(["m1", "m2"]).size()
    print("\n===== 显著性汇总（p<0.05 的配置数 / 总 16 配置）=====")
    print(sig_pairs)
    print("\nSaved significance_v2.csv")
    return df


def smoke():
    old_ep = M.TRAIN_EPISODES
    M.TRAIN_EPISODES = 30
    cfg = dict(CFG_FULL, agents={"PPO", "Fixed", "EqualGran"})
    row, ps = run_one_config(0.85, 1.2, [0], cfg)
    M.TRAIN_EPISODES = old_ep
    print("SMOKE OK row:", {k: row[k] for k in row if row[k] is not None})


def bench():
    old_ep = M.TRAIN_EPISODES
    M.TRAIN_EPISODES = 100
    for name, mk in [("PPO", lambda: M.PPO(M.UavDynamicSemanticCorrFadingEnv(0.85, 1.2), seed=0)),
                     ("DQN", lambda: M.DQN(M.UavDynamicSemanticCorrFadingEnv(0.85, 1.2), seed=0))]:
        ag = mk()
        t0 = time.time()
        ag.train(100)
        dt = time.time() - t0
        print(f"{name}: 100 episodes in {dt:.1f}s -> est full(800)={dt*8/60:.1f}min "
              f"x{16*8} cfgs = {dt*8*16*8/3600:.1f}h (PPO) per agent")
    M.TRAIN_EPISODES = old_ep


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    seeds = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else list(range(8))
    M.ENABLE_CHECKPOINT = True
    if mode == "smoke":
        smoke()
    elif mode == "bench":
        bench()
    elif mode == "main":
        run_main(seeds)
    elif mode == "m3":
        run_m3(seeds)
    elif mode == "m5":
        run_m5(seeds)
    elif mode == "sig":
        run_significance()
    elif mode == "all":
        run_main(seeds)
        run_m3(seeds)
        run_m5(seeds)
        run_significance()
    else:
        print("unknown mode", mode)
