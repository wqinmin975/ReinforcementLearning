# -*- coding: utf-8 -*-
"""P1 §5e: extended-training-budget test (resolves report §6 item 8).

Report §6 item 8 states the curves "have not plateaued at 800 episodes" and that
"whether a longer budget narrows the EqualGran gap is untested". This run tests
exactly that: train the parent PPO to 4x the paper's validated budget
(3200 episodes) on 8 seeds, greedy-evaluating every 400 episodes on a FIXED
eval env, with an EqualGran (global_g=2) reference measured on the same eval set.

Resumability: rows are written atomically as they are produced. On restart, a
seed whose full checkpoint set is present is skipped; a partially recorded seed
is retrained from scratch (torch training state is not reproducible across
processes, so per-seed redo is the safe unit).

Output: exp_results/p1_extended_budget.csv
"""
import os
import time

import pandas as pd

import p1_variants as p

CSV = os.path.join("exp_results", "p1_extended_budget.csv")
RHO, SIGMA = 0.85, 1.2
SEEDS = list(range(8))
TOTAL_EPISODES = 3200          # 4x the paper's validated 800-episode budget
EVAL_EVERY = 400               # checkpoints: 400, 800, ..., 3200
EVAL_EPISODES = 50
CHECKPOINTS = list(range(EVAL_EVERY, TOTAL_EPISODES + 1, EVAL_EVERY))


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
    have = {}
    if len(df) and {"seed", "episode"} <= set(df.columns):
        for s, g in df.groupby("seed"):
            have[int(s)] = set(int(e) for e in g["episode"])
    for s in SEEDS:
        if len(have.get(s, set())) >= len(CHECKPOINTS):
            log(f"seed{s}: complete ({len(have[s])} ckpts), skipping")
    log(f"target: {len(SEEDS)} seeds x {len(CHECKPOINTS)} checkpoints "
        f"(0-{TOTAL_EPISODES} eps, every {EVAL_EVERY})")

    for seed in SEEDS:
        if len(have.get(seed, set())) >= len(CHECKPOINTS):
            continue
        t0 = time.time()
        train_env = p.UavDynamicSemanticCorrFadingEnv(rho=RHO, sigma_est=SIGMA)
        train_env.reset(seed=seed)
        agent = p.PPO(train_env, seed=seed)
        eval_env = p.UavDynamicSemanticCorrFadingEnv(rho=RHO, sigma_est=SIGMA)

        # drop any partial rows for this seed, then rewrite as we go
        if len(df) and (df["seed"] == seed).any():
            df = df[df["seed"] != seed].copy()

        done_eps = 0
        while done_eps < TOTAL_EPISODES:
            agent.train(done_eps + EVAL_EVERY, resume_from=done_eps)
            done_eps += EVAL_EVERY
            agent.env = eval_env
            eval_env.reset(seed=seed)          # identical eval set at every checkpoint
            rec, _, bit, rec_ep, _, _ = agent.evaluate(EVAL_EPISODES)
            agent.env = train_env
            df = pd.concat([df, pd.DataFrame([dict(
                seed=seed, episode=done_eps, ppo_recall=rec,
                ppo_recall_ep=rec_ep, ppo_bits=bit)])], ignore_index=True)
            save_df(df, CSV)
            log(f"  seed{seed} @ {done_eps:5d} eps  recall={rec:.3f} bits={bit:6.1f} "
                f"({time.time() - t0:.0f}s elapsed)")

        # EqualGran reference on the SAME eval set (deterministic baseline, one value per seed)
        eval_env.reset(seed=seed)
        eq = p.EqualGranBaseline(eval_env)
        eq_rec, _, eq_bit, eq_rec_ep, _, _ = eq.evaluate(EVAL_EPISODES, global_g=2)
        df.loc[df["seed"] == seed, "eq_recall"] = eq_rec
        df.loc[df["seed"] == seed, "eq_bits"] = eq_bit
        save_df(df, CSV)
        log(f"seed{seed}: done in {time.time() - t0:.0f}s | EqualGran(reference)={eq_rec:.3f}")

    log("done.")
    print()
    sub_last = df[df.episode == TOTAL_EPISODES]
    pr, er = sub_last["ppo_recall"], sub_last["eq_recall"]
    print(f"PPO @ {TOTAL_EPISODES} eps : {pr.mean():.3f} +- {pr.std(ddof=1):.3f}")
    print(f"EqualGran (reference)  : {er.mean():.3f} +- {er.std(ddof=1):.3f}  "
          f"(gap = {er.mean() - pr.mean():.3f})")
    print("\ncheckpoint trajectory (mean recall over 8 seeds):")
    for ep in CHECKPOINTS:
        s = df[df.episode == ep]
        print(f"  {ep:5d} eps : {s['ppo_recall'].mean():.3f} "
              f"+- {s['ppo_recall'].std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
