# -*- coding: utf-8 -*-
"""
种子离散度检验：检验论文 §6.1 中「PPO 的优化方差高于 off-policy Double-DQN」这一说法。
数据：exp_results/ablation_perseed_v2.csv（16 个 (rho, sigma_est) 条件 x 8 种子）。

方法：
  1) 每个条件内分别计算 PPO / DQN 在 8 个种子上的 Recall@high 样本标准差 (ddof=1) 与方差。
  2) 主检验（pooled Brown-Forsythe / Levene，center='median'）：
     对每个条件、每个方法，取该条件内 8 个种子 Recall@high 的中位数，计算
     d = |y - median|，再把 16 个条件的 d 全部池化为每组 128 个观测，
     对两组 d 做 Levene（等价于对 |y-median| 做单因素 ANOVA）。
     逐条件去中位数后，条件间的均值差异已被消除，池化后比较的纯粹是离散度。
  3) 辅助检验：逐条件 Levene（每组 8 vs 8，功效很低，仅作参考）；
     对 16 个条件的 SD 差值做配对 Wilcoxon 符号秩检验与符号检验。
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

SAVE_DIR = "./exp_results"


def main():
    ps = pd.read_csv(os.path.join(SAVE_DIR, "ablation_perseed_v2.csv"))
    p = ps[ps.method == "PPO"][["rho", "sigma_est", "seed", "recall"]]
    d = ps[ps.method == "DQN"][["rho", "sigma_est", "seed", "recall"]]

    keys = ["rho", "sigma_est"]
    g = p.groupby(keys)["recall"]
    h = d.groupby(keys)["recall"]

    per = pd.DataFrame({
        "PPO_sd": g.std(ddof=1), "PPO_var": g.var(ddof=1), "PPO_mean": g.mean(),
        "DQN_sd": h.std(ddof=1), "DQN_var": h.var(ddof=1), "DQN_mean": h.mean(),
    }).reset_index()
    per["sd_diff"] = per.PPO_sd - per.DQN_sd
    per["sd_ratio"] = per.PPO_sd / per.DQN_sd
    per.to_csv(os.path.join(SAVE_DIR, "levene_ppv_dqn_percond.csv"), index=False)

    print("=== per-condition seed SD of Recall@high (8 seeds each) ===")
    print(per.round(4).to_string(index=False))

    n_cond = len(per)
    n_gt = int((per.sd_diff > 0).sum())
    print(f"\nconditions where PPO SD > DQN SD: {n_gt}/{n_cond}")
    print(f"mean  SD: PPO {per.PPO_sd.mean():.4f} | DQN {per.DQN_sd.mean():.4f} "
          f"| ratio {per.PPO_sd.mean()/per.DQN_sd.mean():.3f}")
    print(f"median SD: PPO {per.PPO_sd.median():.4f} | DQN {per.DQN_sd.median():.4f} "
          f"| ratio {per.PPO_sd.median()/per.DQN_sd.median():.3f}")

    # ---- 主检验：pooled Brown-Forsythe ----
    dev_p, dev_d = [], []
    for _, r in per.iterrows():
        yp = ps[(ps.method == "PPO") & (ps.rho == r.rho) &
                (ps.sigma_est == r.sigma_est)].recall.values
        yd = ps[(ps.method == "DQN") & (ps.rho == r.rho) &
                (ps.sigma_est == r.sigma_est)].recall.values
        dev_p.append(np.abs(yp - np.median(yp)))
        dev_d.append(np.abs(yd - np.median(yd)))
    dev_p = np.concatenate(dev_p)
    dev_d = np.concatenate(dev_d)
    # 注意：dev_* 已经是「逐条件中位数中心化后的绝对偏差」。
    # 不能再把它们喂给 scipy.stats.levene(center='median')——那会二次中心化。
    # 池化 Brown-Forsythe 的正确形式就是：对这批 |y - median_i| 做单因素 ANOVA。
    F, pval = stats.f_oneway(dev_p, dev_d)
    print(f"\n=== pooled Brown-Forsythe (median-centred |dev|, one-way ANOVA) ===")
    print(f"F(1, {len(dev_p) + len(dev_d) - 2}) = {F:.4f}, p = {pval:.5f}, "
          f"n = {len(dev_p)} vs {len(dev_d)}")
    print(f"mean |dev|: PPO {dev_p.mean():.4f} | DQN {dev_d.mean():.4f} "
          f"| ratio {dev_p.mean()/dev_d.mean():.3f}")

    # 敏感性分析：改用逐条件「均值」中心化的绝对偏差
    mdev_p, mdev_d = [], []
    for _, r in per.iterrows():
        yp = ps[(ps.method == "PPO") & (ps.rho == r.rho) &
                (ps.sigma_est == r.sigma_est)].recall.values
        yd = ps[(ps.method == "DQN") & (ps.rho == r.rho) &
                (ps.sigma_est == r.sigma_est)].recall.values
        mdev_p.append(np.abs(yp - yp.mean()))
        mdev_d.append(np.abs(yd - yd.mean()))
    mdev_p, mdev_d = np.concatenate(mdev_p), np.concatenate(mdev_d)
    Fm, pm = stats.f_oneway(mdev_p, mdev_d)
    print(f"[sensitivity] mean-centred |dev|: F(1, {len(mdev_p)+len(mdev_d)-2}) = {Fm:.4f}, "
          f"p = {pm:.5f} | mean |dev| ratio {mdev_p.mean()/mdev_d.mean():.3f}")

    # ---- 辅助：逐条件 Levene（8 vs 8，低功效） ----
    sig_rows = []
    for _, r in per.iterrows():
        yp = ps[(ps.method == "PPO") & (ps.rho == r.rho) &
                (ps.sigma_est == r.sigma_est)].recall.values
        yd = ps[(ps.method == "DQN") & (ps.rho == r.rho) &
                (ps.sigma_est == r.sigma_est)].recall.values
        w, pv = stats.levene(yp, yd, center="median")
        sig_rows.append(dict(rho=r.rho, sigma_est=r.sigma_est, W=w, p=pv))
    sig = pd.DataFrame(sig_rows)
    print(f"\n[auxiliary] per-condition Levene: significant at 0.05 in "
          f"{int((sig.p < 0.05).sum())}/{n_cond} conditions "
          f"(min p = {sig.p.min():.4f}); NOTE 8 vs 8 gives very low power.")

    # ---- 辅助：对 16 个条件的 SD 做配对检验 ----
    try:
        ws, pws = stats.wilcoxon(per.PPO_sd, per.DQN_sd)
    except Exception as e:
        ws, pws = np.nan, np.nan
        print("wilcoxon failed:", e)
    bt, pbt = stats.ttest_rel(per.PPO_sd, per.DQN_sd)
    npos = int((per.sd_diff > 0).sum())
    psign = stats.binomtest(npos, n_cond, 0.5).pvalue
    print(f"\n[auxiliary] paired on 16 condition SDs: Wilcoxon W = {ws:.1f}, p = {pws:.5f}")
    print(f"[auxiliary] paired t-test on SDs: t = {bt:.4f}, p = {pbt:.5f}")
    print(f"[auxiliary] sign test: {npos}/{n_cond} positive, p = {psign:.5f}")

    out = pd.DataFrame([dict(
        n_conditions=n_cond,
        PPO_sd_mean=per.PPO_sd.mean(), PPO_sd_median=per.PPO_sd.median(),
        DQN_sd_mean=per.DQN_sd.mean(), DQN_sd_median=per.DQN_sd.median(),
        sd_ratio_mean=per.PPO_sd.mean() / per.DQN_sd.mean(),
        sd_ratio_median=per.PPO_sd.median() / per.DQN_sd.median(),
        n_cond_PPO_sd_greater=n_gt,
        bf_F_median=F, bf_p_median=pval,
        bf_F_mean=Fm, bf_p_mean=pm,
        pooled_n_per_group=len(dev_p),
        wilcoxon_p=pws, paired_t_p=pbt, sign_test_p=psign,
        percond_levene_sig=int((sig.p < 0.05).sum()),
    )])
    out.to_csv(os.path.join(SAVE_DIR, "levene_ppv_dqn.csv"), index=False)
    print("\n=== summary row ===")
    print(out.round(5).to_string(index=False))

    print("\n=== check vs paper's 2x claim (training-curve SD 0.050 vs 0.024, ratio 2.08) ===")
    print(f"observed mean-SD ratio = {per.PPO_sd.mean()/per.DQN_sd.mean():.3f}, "
          f"median-SD ratio = {per.PPO_sd.median()/per.DQN_sd.median():.3f}, "
          f"mean |dev| ratio = {dev_p.mean()/dev_d.mean():.3f}")


if __name__ == "__main__":
    main()
