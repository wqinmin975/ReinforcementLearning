# -*- coding: utf-8 -*-
"""
P1 FULL-baseline driver (resumable). Runs three stages, appending one row at a time to
per-stage CSVs and skipping already-completed rows on restart, so a multi-hour run can be
interrupted and resumed without losing completed work.

  Stage 1  exp_results/p1_hparam_sweep_ppo.csv   16 configs x 8 seeds, PPO, 800 eps
  Stage 2  exp_results/p1_training_curves.csv     PPO+DQN x 8 seeds, 800 eps, eval every 25
           exp_results/p1_training_curves.png
  Stage 3  exp_results/p1_dp_benchmark.csv        3 envs x 8 seeds, DP myopic oracle

Reference env: rho=0.85, sigma_est=1.2. Does NOT modify uav_semantic_multi_seed_ablation.py.

Usage:
    python run_p1_full.py            # run all three stages (skips completed rows)
    python run_p1_full.py stage1     # run only stage 1
    python run_p1_full.py summary    # print summaries from the CSVs already on disk
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd

import p1_variants as p

SWEEP_CSV = os.path.join("exp_results", "p1_hparam_sweep_ppo.csv")
CURVES_CSV = os.path.join("exp_results", "p1_training_curves.csv")
CURVES_PNG = os.path.join("exp_results", "p1_training_curves.png")
CURVES_META = os.path.join("exp_results", "p1_training_curves_meta.json")
DP_CSV = os.path.join("exp_results", "p1_dp_benchmark.csv")

RHO, SIGMA = 0.85, 1.2
SEEDS = list(range(8))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_df(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            log(f"WARN: could not read {path} ({e}); starting empty")
    return pd.DataFrame()


def save_df(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)  # atomic replace -> crash-safe


# ------------------------------------------------------------------ Stage 1
def stage1(train_episodes=800, eval_episodes=50):
    configs = p.make_hparam_configs_representative()
    df = load_df(SWEEP_CSV)
    done = set()
    if "config_id" in df and "seed" in df:
        done = set(zip(df["config_id"].astype(int), df["seed"].astype(int)))
    log(f"Stage1 start: {len(done)} rows already done (target {len(configs)*len(SEEDS)})")
    for ci, cfg in enumerate(configs):
        for seed in SEEDS:
            if (ci, seed) in done:
                continue
            t0 = time.time()
            env = p.UavDynamicSemanticCorrFadingEnv(rho=RHO, sigma_est=SIGMA)
            env.reset(seed=seed)
            agent = p.PPOConfigurable(env, seed=seed, gamma=cfg["gamma"],
                                      gae_lambda=cfg["gae_lambda"], ent_coef=cfg["ent_coef"],
                                      clip=cfg["clip"], hidden=cfg["hidden"])
            agent.train(train_episodes)
            env.reset(seed=seed)
            rec, _, bit, rec_ep, _, _ = agent.evaluate(eval_episodes)
            row = dict(config_id=ci, gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"],
                       ent_coef=cfg["ent_coef"], clip=cfg["clip"],
                       hidden=str(cfg["hidden"]), seed=seed,
                       recall=rec, recall_ep=rec_ep, bits=bit)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            save_df(df, SWEEP_CSV)
            log(f"Stage1 cfg{ci:2d} seed{seed} recall={rec:.3f} bits={bit:7.1f} "
                f"({time.time()-t0:.0f}s)  [g={cfg['gamma']} lam={cfg['gae_lambda']} "
                f"ent={cfg['ent_coef']} clip={cfg['clip']} h={cfg['hidden']}]")
    log("Stage1 done.")


# ------------------------------------------------------------------ Stage 2
def _plot_curves(df, eq_mean, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for agent_name, color in [("PPO", "C0"), ("DQN", "C1")]:
        sub = df[df["agent"] == agent_name]
        g = sub.groupby("episode")["recall_final"]
        mean, std = g.mean(), g.std(ddof=1)
        eps = mean.index.values
        ax.plot(eps, mean.values, marker="o", ms=3, lw=1.4, label=agent_name, color=color)
        ax.fill_between(eps, (mean - std).values, (mean + std).values, alpha=0.2, color=color)
    ax.axhline(eq_mean, ls="--", color="k", lw=1.2, label=f"EqualGran ({eq_mean:.3f})")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("High-Priority Recall@high")
    ax.set_title(f"Training curves (rho={RHO}, sigma_est={SIGMA}), mean+-std over {len(SEEDS)} seeds")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def stage2(train_episodes=800, eval_every=25, eval_episodes=50):
    n_ep = train_episodes // eval_every  # expected eval points per (agent, seed)

    # EqualGran reference (computed once, cached)
    if os.path.exists(CURVES_META):
        eq_mean = json.load(open(CURVES_META))["eq_mean"]
        log(f"Stage2 EqualGran reference (cached) = {eq_mean:.4f}")
    else:
        eq_recs = []
        for seed in SEEDS:
            e = p.UavDynamicSemanticCorrFadingEnv(rho=RHO, sigma_est=SIGMA)
            e.reset(seed=seed)
            eq = p.EqualGranBaseline(e)
            eq_recs.append(eq.evaluate(eval_episodes, global_g=2)[0])
        eq_mean = float(np.mean(eq_recs))
        os.makedirs(os.path.dirname(CURVES_META), exist_ok=True)
        json.dump({"eq_mean": eq_mean}, open(CURVES_META, "w"))
        log(f"Stage2 EqualGran reference = {eq_mean:.4f} (seeds mean, std={np.std(eq_recs, ddof=1):.4f})")

    df = load_df(CURVES_CSV)
    for seed in SEEDS:
        for agent_name, AgentCls in [("PPO", p.PPO), ("DQN", p.DQN)]:
            have = df[(df["agent"] == agent_name) & (df["seed"] == seed)] if len(df) else pd.DataFrame()
            if len(have) >= n_ep:
                continue  # (agent, seed) already complete -> skip (atomic granularity)
            t0 = time.time()
            train_env = p.UavDynamicSemanticCorrFadingEnv(rho=RHO, sigma_est=SIGMA)
            train_env.reset(seed=seed)
            agent = AgentCls(train_env, seed=seed)
            eval_env = p.UavDynamicSemanticCorrFadingEnv(rho=RHO, sigma_est=SIGMA)
            new_rows, done_eps = [], 0
            while done_eps < train_episodes:
                agent.train(done_eps + eval_every, resume_from=done_eps)
                done_eps += eval_every
                agent.env = eval_env
                eval_env.reset(seed=seed)          # identical eval set each eval point
                rec, sem, bit, rec_ep, _, _ = agent.evaluate(eval_episodes)
                agent.env = train_env
                new_rows.append(dict(agent=agent_name, seed=seed, episode=done_eps,
                                     recall_final=rec, recall_perstep=rec_ep,
                                     bits=bit, sem_utility=sem))
            # only persist once the whole (agent, seed) is complete (avoids partial/duplicate rows)
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            save_df(df, CURVES_CSV)
            log(f"Stage2 {agent_name} seed{seed} done (800 eps, {n_ep} eval pts) "
                f"final_recall={new_rows[-1]['recall_final']:.3f} ({time.time()-t0:.0f}s)")

    # regenerate PNG once curves are complete
    complete = all(
        (len(df[(df['agent'] == a) & (df['seed'] == s)]) >= n_ep)
        for a in ("PPO", "DQN") for s in SEEDS
    )
    if complete and not os.path.exists(CURVES_PNG):
        _plot_curves(df, eq_mean, CURVES_PNG)
        log(f"Stage2 PNG written: {CURVES_PNG}")
    log("Stage2 done.")


# ------------------------------------------------------------------ Stage 3
def stage3(max_steps=200):
    envs = [("Original", p.UavDynamicSemanticCorrFadingEnv),
            ("AlHourani", p.EnvAlHourani),
            ("ComplexCSI", p.EnvComplexCSI)]
    df = load_df(DP_CSV)
    done = set()
    if "env" in df and "seed" in df:
        done = set(zip(df["env"].astype(str), df["seed"].astype(int)))
    for ename, ECls in envs:
        for seed in SEEDS:
            if (ename, seed) in done:
                continue
            env = ECls(rho=RHO, sigma_est=SIGMA)
            dp = p.DPBaseline(env)
            stats = dp.greedy_rollout_stats(seed=seed, max_steps=max_steps)
            row = dict(env=ename, seed=seed, **stats)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            save_df(df, DP_CSV)
            log(f"Stage3 {ename:10s} seed{seed} recall_final={stats['recall_final']:.3f} "
                f"recall_ps={stats['recall_per_step']:.3f} bits={stats['mean_bits']:.1f} "
                f"len={stats['episode_length']}")
    log("Stage3 done.")


# ------------------------------------------------------------------ Summary
def summary():
    pd.set_option("display.width", 200)
    if os.path.exists(SWEEP_CSV):
        s = pd.read_csv(SWEEP_CSV)
        print("\n===== Stage1 sweep (PPO, 8 seeds) recall mean±std per config =====")
        agg = s.groupby(["config_id", "gamma", "gae_lambda", "ent_coef", "clip", "hidden"])\
               .agg(recall_mean=("recall", "mean"), recall_std=("recall", lambda x: np.std(x, ddof=1)),
                    bits_mean=("bits", "mean")).reset_index().sort_values("recall_mean", ascending=False)
        print(agg.to_string(index=False))
        base = s[s.config_id == 0]["recall"].mean()
        print(f"baseline(config0) mean recall = {base:.4f}")
    if os.path.exists(CURVES_CSV):
        c = pd.read_csv(CURVES_CSV)
        print("\n===== Stage2 curves: final-episode (800) recall across seeds =====")
        fin = c[c.episode == c.episode.max()].groupby("agent")["recall_final"]\
              .agg(["mean", lambda x: np.std(x, ddof=1)])
        print(fin.to_string())
        meta = json.load(open(CURVES_META)) if os.path.exists(CURVES_META) else {}
        print(f"EqualGran reference recall = {meta.get('eq_mean', float('nan')):.4f}")
    if os.path.exists(DP_CSV):
        d = pd.read_csv(DP_CSV)
        print("\n===== Stage3 DP myopic oracle (8 seeds) per env =====")
        print(d.groupby("env").agg(recall_final=("recall_final", "mean"),
                                   recall_per_step=("recall_per_step", "mean"),
                                   mean_bits=("mean_bits", "mean"),
                                   ep_len=("episode_length", "mean")).to_string())


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t_all = time.time()
    if which in ("all", "stage1"):
        t = time.time(); stage1(); log(f"Stage1 wall time {time.time()-t:.0f}s")
    if which in ("all", "stage2"):
        t = time.time(); stage2(); log(f"Stage2 wall time {time.time()-t:.0f}s")
    if which in ("all", "stage3"):
        t = time.time(); stage3(); log(f"Stage3 wall time {time.time()-t:.0f}s")
    if which == "summary":
        summary()
    log(f"ALL DONE in {time.time()-t_all:.0f}s")
    summary()


if __name__ == "__main__":
    main()
