# -*- coding: utf-8 -*-
"""P1 §5d: per-channel negative-result confirmation (resumable).

For each channel env (Original / AlHourani / ComplexCSI) and each seed 0-7:
  - train the parent PPO for 800 episodes (the paper's validated protocol),
    greedy-evaluate 50 episodes;
  - EqualGran (global_g=2) greedy-evaluate 50 episodes on the same env/seed.
Writes one row per (env, seed) atomically; skips completed rows on restart.
Output: exp_results/p1_channel_confirm.csv
"""
import os
import time

import pandas as pd

import p1_variants as p

CSV = os.path.join("exp_results", "p1_channel_confirm.csv")
RHO, SIGMA = 0.85, 1.2
SEEDS = list(range(8))
ENVS = [("Original", p.UavDynamicSemanticCorrFadingEnv),
        ("AlHourani", p.EnvAlHourani),
        ("ComplexCSI", p.EnvComplexCSI)]


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


def main():
    df = load_df(CSV)
    done = set()
    if "env" in df and "seed" in df:
        done = set(zip(df["env"].astype(str), df["seed"].astype(int)))
    log(f"start: {len(done)} rows done (target {len(ENVS) * len(SEEDS)})")
    for ename, ECls in ENVS:
        for seed in SEEDS:
            if (ename, seed) in done:
                continue
            t0 = time.time()
            env = ECls(rho=RHO, sigma_est=SIGMA)
            env.reset(seed=seed)
            agent = p.PPO(env, seed=seed)
            agent.train(800)
            env.reset(seed=seed)
            ppo_rec, _, ppo_bit, ppo_rec_ep, _, _ = agent.evaluate(50)
            env.reset(seed=seed)
            eq = p.EqualGranBaseline(env)
            eq_rec, _, eq_bit, eq_rec_ep, _, _ = eq.evaluate(50, global_g=2)
            row = dict(env=ename, seed=seed,
                       ppo_recall=ppo_rec, ppo_recall_ep=ppo_rec_ep, ppo_bits=ppo_bit,
                       eq_recall=eq_rec, eq_recall_ep=eq_rec_ep, eq_bits=eq_bit)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            save_df(df, CSV)
            log(f"{ename:10s} seed{seed}  PPO={ppo_rec:.3f} bits={ppo_bit:6.1f}  "
                f"EqualGran={eq_rec:.3f} bits={eq_bit:6.1f}  ({time.time() - t0:.0f}s)")
    log("done.")
    print()
    for ename, _ in ENVS:
        sub = df[df.env == ename]
        pr, er = sub["ppo_recall"], sub["eq_recall"]
        flag = "PPO BELOW (negative result holds)" if pr.mean() < er.mean() else "PPO ABOVE (FLAG: check)"
        print(f"{ename:10s}: PPO {pr.mean():.3f}+-{pr.std(ddof=1):.3f}  "
              f"EqualGran {er.mean():.3f}+-{er.std(ddof=1):.3f}  -> {flag}")


if __name__ == "__main__":
    main()
