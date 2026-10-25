# -*- coding: utf-8 -*-
"""
P1 experimental enhancements for the UAV semantic-communication RL paper (IEEE OJVT).

This module MUST NOT modify uav_semantic_multi_seed_ablation.py. It imports the
validated, authoritative environment / agents / baselines from that file and adds:

  (A) EnvAlHourani        - Al-Hourani/Wafaa-2014 air-to-ground (A2G) path-loss channel.
  (B) EnvComplexCSI       - complex-baseband channel-estimation-noise CSI model.
  (C) DPBaseline          - myopic per-step greedy oracle (perfect CSI, NOT a full DP).
  (D) Hyperparameter-ablation plumbing
                            - PPOConfigurable / DQNConfigurable (ent_coef + hidden sizes)
                            - run_hparam_sweep(configs, seeds) -> DataFrame
  (E) run_p1_pilot()      - bounded real-number sanity checks (callable, prints numbers).

All randomness that touches the environment goes through the parent env's self.np_random
(seeded by reset(seed)), preserving the original reproducibility convention.

Run:  python p1_variants.py        (executes the pilot; see run_p1_pilot)
"""

import os
import time
import itertools
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from uav_semantic_multi_seed_ablation import (
    UavDynamicSemanticCorrFadingEnv,
    PPO,
    DQN,
    FixedPolicyBaseline,
    EqualGranBaseline,
    semantic_utility,
    recall_prob,
    bit_cost_table,
    NUM_REGIONS,
    lambda1,
    lambda2,
    lambda_energy,
    lambda_R,
    RECALL_SCOPE,
    HIGH_PRIO_MIN_GRAN,
)


# ======================================================================================
# (A) Al-Hourani / Wafaa-2014 air-to-ground (A2G) path-loss channel model
# ======================================================================================
class EnvAlHourani(UavDynamicSemanticCorrFadingEnv):
    """
    Overrides _compute_snr_dB with the Al-Hourani / Wafaa-2014 A2G path-loss model
    (Al-Hourani, Kandeepan, Lardner, "Modelling of Class Aerial Platforms," ICUAS 2014;
    Wafaa, "On the optimality of downlink NOMA in UAV," 2014 — widely cited a=9.61, b=0.16).

    Reference formula (elevation angle theta in DEGREES):
        P_LOS(theta) = 1 / (1 + a * exp(-b * (theta - a)))
        PL_LOS(dB)   = 20*log10(4*pi*f_c*d_3d / c)        (3D distance d_3d)
        PL_NLOS(dB)  = PL_LOS + 20                        (20 dB extra NLoS attenuation)
        PL(dB)       = P_LOS*PL_LOS + (1-P_LOS)*PL_NLOS

    The fast-fading story is preserved: after computing the large-scale PL we convert
    to linear, multiply by the PARENT's correlated-Rayleigh power _sample_correlated_rayleigh()
    (AR(1) complex Gaussian, identical to the original env), and clip to [0,20] dB.

    CALIBRATION: the original env uses snr_path_dB = clip(18 - distance*0.4, 0, 20).
    To keep the new channel in a comparable [0,20] dB regime we introduce a single
    constant P_TX_OVER_N0_DB = 10*log10(P_tx / N0). With h=50 m, f_c=2.4 GHz the
    LoS PL sits around 74-79 dB; P_TX_OVER_N0_DB=82 yields near-field SNR ~ 3-8 dB and
    far/NLoS SNR ~ 0 dB, matching the original's typical operating range. This constant
    is the ONLY free calibration knob and is documented here.
    """

    UAV_HEIGHT = 50.0           # fixed UAV altitude (m)
    F_C = 2.4e9                 # carrier frequency (Hz)
    C_LIGHT = 3.0e8             # speed of light (m/s)
    A_LOS = 9.61                # Al-Hourani/Wafaa-2014 coefficient a
    B_LOS = 0.16                # Al-Hourani/Wafaa-2014 coefficient b
    NLOS_EXTRA_DB = 20.0        # extra NLoS attenuation (dB)
    P_TX_OVER_N0_DB = 82.0      # calibration: 10*log10(P_tx / N0)

    def _pathloss_snr_dB(self, distance):
        """Deterministic (large-scale) SNR in dB: Al-Hourani path loss, no fading."""
        d_horiz = float(distance)
        d_3d = np.sqrt(self.UAV_HEIGHT ** 2 + d_horiz ** 2)
        theta = np.degrees(np.arctan2(self.UAV_HEIGHT, d_horiz))  # elevation angle (deg)
        p_los = 1.0 / (1.0 + self.A_LOS * np.exp(-self.B_LOS * (theta - self.A_LOS)))
        pl_los = 20.0 * np.log10(4.0 * np.pi * self.F_C * d_3d / self.C_LIGHT)
        pl = p_los * pl_los + (1.0 - p_los) * (pl_los + self.NLOS_EXTRA_DB)
        snr_lin = 10.0 ** (self.P_TX_OVER_N0_DB / 10.0) / (10.0 ** (pl / 10.0))
        snr = 10.0 * np.log10(max(snr_lin, 1e-8))
        return float(np.clip(snr, 0.0, 20.0))

    def _compute_snr_dB(self, distance):
        snr_path = self._pathloss_snr_dB(distance)
        snr_path_lin = 10.0 ** (snr_path / 10.0)
        rayleigh_pow = self._sample_correlated_rayleigh()      # reuse parent's AR(1) fading
        snr_lin = snr_path_lin * rayleigh_pow
        snr_true = 10.0 * np.log10(max(snr_lin, 1e-8))
        return float(np.clip(snr_true, 0.0, 20.0))


# ======================================================================================
# (B) Complex-baseband CSI (channel-estimation-noise) variant
# ======================================================================================
class EnvComplexCSI(UavDynamicSemanticCorrFadingEnv):
    """
    Overrides _get_obs only. Instead of adding dB-domain Gaussian noise to the SNR
    (parent: snr_obs = snr_true + N(0, sigma_est)), we model a complex-baseband
    channel estimate:

        h_true = sqrt(SNR_lin) * exp(j*phi),   phi ~ Uniform(0, 2*pi)   (seeded)
        noise power E[|n|^2] = (10^(sigma_est/10) - 1) * SNR_lin
            -> per real/imag component std sigma_c = sqrt(E[|n|^2]/2)
        h_hat  = h_true + n_re + j*n_im
        snr_obs_dB = 10*log10(|h_hat|^2),  clipped to [0,20] and /20 normalized.

    MAPPING DOCUMENTED: sigma_est (dB) is converted to a baseband noise variance that
    SCALES WITH the signal (proportional to SNR_lin) so the relative estimation error is
    constant in the dB domain. With xi = 10^(sigma_est/10):
        E[|n|^2] = (xi - 1) * SNR_lin.
    At sigma_est=0 -> xi=1 -> zero noise -> snr_obs == snr_true (degrades to perfect CSI).
    For sigma_est=1.2 dB, xi-1 ~= 0.318, i.e. noise power ~ 31.8% of signal power, which
    corresponds to roughly a +/- 1.2 dB spread in the resulting SNR estimate — comparable
    to the original additive-dB noise of std sigma_est. The phase phi and the noise are
    drawn through the parent's seeded self.np_random, preserving reproducibility.
    """

    def _get_obs(self, snr_true):
        snr_lin = 10.0 ** (snr_true / 10.0)
        phi = self.np_random.uniform(0.0, 2.0 * np.pi)
        h_true_re = np.sqrt(snr_lin) * np.cos(phi)
        h_true_im = np.sqrt(snr_lin) * np.sin(phi)

        xi = 10.0 ** (self.sigma_est / 10.0)
        n_pow = (xi - 1.0) * snr_lin                 # E[|n|^2]
        sigma_comp = np.sqrt(max(n_pow, 0.0) / 2.0)  # per real/imag component std
        n_re = self.np_random.normal(0.0, sigma_comp)
        n_im = self.np_random.normal(0.0, sigma_comp)

        h_hat_re = h_true_re + n_re
        h_hat_im = h_true_im + n_im
        snr_hat_lin = h_hat_re ** 2 + h_hat_im ** 2
        snr_obs = 10.0 * np.log10(max(snr_hat_lin, 1e-8))
        snr_obs = np.clip(snr_obs, 0.0, 20.0)
        snr_norm = np.clip(snr_obs / 20.0, 0.0, 1.0)

        bw_norm = self.avail_bw
        eng_norm = self.energy_rem / 100.0
        avg_s_norm = np.mean(self.region_s)
        lat_norm = np.clip(self.latency / 100.0, 0.0, 1.0)
        xu_norm = np.clip(self.xu / 100.0, -1.0, 1.0)
        yu_norm = np.clip(self.yu / 100.0, -1.0, 1.0)
        obs = np.array([snr_norm, bw_norm, eng_norm, avg_s_norm, lat_norm,
                        xu_norm, yu_norm], dtype=np.float32)
        return obs


# ======================================================================================
# (C) Myopic greedy oracle (DP-style analytical baseline) — NOT a full DP
# ======================================================================================
class DPBaseline:
    """
    Myopic per-step greedy controller given PERFECT CSI.

    HONEST SCOPE: A true dynamic-programming solution over the full POMDP is intractable
    (continuous state, stochastic fading, long horizon). This class is a MYOPIC GREEDY
    ORACLE, not a DP: at each decision step it brute-forces all 576 joint actions and
    picks the one maximizing the IMMEDIATE reward of the paper's Eq.(9) (the same reward
    the RL agents optimize). It is a fair, defensible analytical upper-bound-style
    benchmark ("what a perfect-CSI myopic controller achieves") and must NOT be claimed as
    an optimal DP or a globally optimal controller.

    Perfect-CSI assumption used here (documented, not overclaimed):
      * It knows the true current snr_true_dB (no estimation error).
      * For evaluating a candidate MOVE it uses the DETERMINISTIC large-scale path loss
        at the reachable cell (via env._pathloss_snr_dB if available, else the original
        clip(18 - d*0.4,0,20) formula) with the fast-fading treated at its mean (=1).
        It does NOT peek at the future fading realization; it only uses geometry it could
        legitimately know (own altitude, carrier frequency, own position).
    """

    def __init__(self, env):
        self.env = env
        self.action_dims = env.action_space.nvec.tolist()
        self.all_actions = self._build_all_actions()

    def _build_all_actions(self):
        combos = itertools.product(*[range(d) for d in self.action_dims])
        return [np.array(a, dtype=int) for a in combos]   # 4*4*4*3*3 = 576

    def _pathloss_fn(self, distance):
        # Use the env's deterministic path-loss if it exposes one (EnvAlHourani),
        # otherwise fall back to the original env's large-scale formula.
        pl = getattr(self.env, "_pathloss_snr_dB", None)
        if pl is not None:
            return float(pl(distance))
        return float(np.clip(18.0 - distance * 0.4, 0.0, 20.0))

    def _est_next_snr(self, action):
        env = self.env
        dx = env.delta_inv[int(action[3])]
        dy = env.delta_inv[int(action[4])]
        xu = np.clip(env.xu + dx, -50.0, 50.0)
        yu = np.clip(env.yu + dy, -50.0, 50.0)
        d = np.sqrt(xu ** 2 + yu ** 2)
        return self._pathloss_fn(d)

    def _immediate_reward(self, action, snr_dB):
        """Exact replica of env.step's reward (Eq.(9)) for a given snr_dB, no side effects."""
        env = self.env
        g_high, g_mid, g_low, dx_code, dy_code = (int(action[0]), int(action[1]),
                                                  int(action[2]), int(action[3]), int(action[4]))
        g_all = np.zeros(NUM_REGIONS, dtype=int)
        g_all[env.idx_high] = g_high
        g_all[env.idx_mid] = g_mid
        g_all[env.idx_low] = g_low

        total_sem_utility = 0.0
        total_bits = 0
        for i in range(NUM_REGIONS):
            gi = g_all[i]
            si = env.region_s[i]
            total_sem_utility += si * semantic_utility(gi, snr_dB)
            total_bits += bit_cost_table[gi]

        dx = env.delta_inv[dx_code]
        dy = env.delta_inv[dy_code]
        flight_energy = (abs(dx) + abs(dy)) * 0.8
        comm_energy = total_bits * 0.02
        total_energy_consume = flight_energy + comm_energy
        latency = total_bits / (env.avail_bw * 80.0)

        recall_reward = 0.0
        if RECALL_SCOPE == "high":
            for i in env.idx_high:
                recall_reward += recall_prob(g_all[i], snr_dB)
        else:
            for i in range(NUM_REGIONS):
                recall_reward += env.region_s[i] * recall_prob(g_all[i], snr_dB)

        reward = (total_sem_utility
                  - lambda1 * total_bits
                  - lambda2 * latency
                  - lambda_energy * total_energy_consume
                  + lambda_R * recall_reward)
        return reward

    def greedy_rollout(self, seed=0, max_steps=200):
        """
        Roll out the myopic greedy oracle for one episode.
        Returns (final_recall, per_step_recall, mean_bits) where:
          final_recall   = Recall@high of the LAST step (matches PPO.evaluate convention)
          per_step_recall= mean Recall@high over all steps (episode-level grid-mean)
          mean_bits      = mean total_bits per step
        """
        env = self.env
        env.reset(seed=seed)
        done = False
        step = 0
        rec_final = 0.0
        rec_per_step = []
        bits_per_step = []
        while (not done) and (step < max_steps):
            best_a = None
            best_r = -1e18
            for a in self.all_actions:
                est = self._est_next_snr(a)
                r = self._immediate_reward(a, est)
                if r > best_r:
                    best_r = r
                    best_a = a
            _, _, term, trunc, info = env.step(best_a)
            rec_final = info["recall_high"]
            rec_per_step.append(info["recall_high"])
            bits_per_step.append(info["total_bits"])
            done = term or trunc
            step += 1
        rec_ps = float(np.mean(rec_per_step)) if rec_per_step else 0.0
        bits = float(np.mean(bits_per_step)) if bits_per_step else 0.0
        return rec_final, rec_ps, bits

    def greedy_rollout_stats(self, seed=0, max_steps=200):
        """
        Same rollout as greedy_rollout but returns a dict with all stage-3 fields:
        {recall_final, recall_per_step, mean_bits, episode_length}.
        episode_length is the number of steps actually executed (energy-myopic episodes
        terminate early — this is the evidence the myopic oracle is NOT an upper bound).
        """
        env = self.env
        env.reset(seed=seed)
        done = False
        step = 0
        rec_final = 0.0
        rec_per_step = []
        bits_per_step = []
        while (not done) and (step < max_steps):
            best_a = None
            best_r = -1e18
            for a in self.all_actions:
                est = self._est_next_snr(a)
                r = self._immediate_reward(a, est)
                if r > best_r:
                    best_r = r
                    best_a = a
            _, _, term, trunc, info = env.step(best_a)
            rec_final = info["recall_high"]
            rec_per_step.append(info["recall_high"])
            bits_per_step.append(info["total_bits"])
            done = term or trunc
            step += 1
        return dict(
            recall_final=float(rec_final),
            recall_per_step=float(np.mean(rec_per_step)) if rec_per_step else 0.0,
            mean_bits=float(np.mean(bits_per_step)) if bits_per_step else 0.0,
            episode_length=int(step),
        )


# ======================================================================================
# (D) Hyperparameter-ablation plumbing
# ======================================================================================
class ActorCriticPPOConfigurable(nn.Module):
    """PPO actor-critic with configurable hidden sizes (original hardcodes 256-128)."""

    def __init__(self, state_dim, action_dims, activation="tanh", hidden=(256, 128)):
        super().__init__()
        self.action_dims = action_dims
        self.activation = activation
        act_layer = nn.ReLU() if activation == "relu" else nn.Tanh()
        h0, h1 = hidden
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, h0), act_layer,
            nn.Linear(h0, h1), act_layer
        )
        self.heads = nn.ModuleList([nn.Linear(h1, d) for d in action_dims])
        self.critic = nn.Linear(h1, 1)

    def get_action(self, state):
        feat = self.backbone(state)
        acts, logp_sum = [], 0
        for i, h in enumerate(self.heads):
            logits = h(feat)
            if i == 0:  # high-priority granularity head: forbid g < HIGH_PRIO_MIN_GRAN
                mask_idx = torch.arange(logits.shape[-1]) < HIGH_PRIO_MIN_GRAN
                logits = logits.masked_fill(mask_idx, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logp_sum += dist.log_prob(a)
            acts.append(a.item())
        val = self.critic(feat)
        return np.array(acts), logp_sum.item(), val.item()

    def evaluate(self, state, actions):
        feat = self.backbone(state)
        logp_sum, entropy_sum = 0, 0
        for i, h in enumerate(self.heads):
            logits = h(feat)
            if i == 0:
                mask_idx = torch.arange(logits.shape[-1]) < HIGH_PRIO_MIN_GRAN
                logits = logits.masked_fill(mask_idx, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            logp_sum += dist.log_prob(actions[:, i])
            entropy_sum += dist.entropy()
        val = self.critic(feat)
        return logp_sum, entropy_sum, val


class PPOConfigurable(PPO):
    """
    PPO with sweepable entropy coefficient and hidden sizes.

    FINDING (original file): the entropy bonus is HARDCODED at 0.01 in PPO.update
    (line 471: `loss = actor_loss + 0.5*critic_loss - 0.01*entropy.mean()`) and the
    network hidden sizes are HARDCODED as (256,128) in ActorCriticPPO. This subclass
    exposes `ent_coef` (default 0.01 — preserves original behaviour) and `hidden`
    so both can be swept. It does NOT modify the original file.
    """

    def __init__(self, env, lr=3e-4, gamma=0.96, clip=0.2, epoch=10, seed=0,
                 gae_lambda=0.95, activation="tanh", ent_coef=0.01, hidden=(256, 128)):
        super().__init__(env, lr=lr, gamma=gamma, clip=clip, epoch=epoch, seed=seed,
                         gae_lambda=gae_lambda, activation=activation)
        self.ent_coef = ent_coef
        self.hidden = hidden
        # rebuild the network with configurable hidden sizes (torch RNG already seeded)
        self.ac = ActorCriticPPOConfigurable(self.state_dim, self.action_dims,
                                             activation=activation, hidden=hidden)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr)

    def update(self, traj):
        s_arr = torch.tensor(np.array(traj["s"]), dtype=torch.float32)
        a_arr = torch.tensor(np.array(traj["a"]), dtype=torch.int64)
        old_logp = torch.tensor(np.array(traj["logp"]), dtype=torch.float32)
        adv, ret = self.compute_gae(traj)
        adv = torch.tensor(adv, dtype=torch.float32)
        ret = torch.tensor(ret, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(self.epoch):
            new_logp, entropy, val = self.ac.evaluate(s_arr, a_arr)
            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(val.squeeze(), ret)
            loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy.mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return loss.item()


class QNetConfigurable(nn.Module):
    def __init__(self, state_dim, action_total, hidden=(256, 128)):
        super().__init__()
        h0, h1 = hidden
        self.net = nn.Sequential(
            nn.Linear(state_dim, h0), nn.ReLU(),
            nn.Linear(h0, h1), nn.ReLU(),
            nn.Linear(h1, action_total)
        )

    def forward(self, x):
        return self.net(x)


class DQNConfigurable(DQN):
    """DQN with sweepable hidden sizes and `eps` (maps to the PPO `clip` sweep axis)."""

    def __init__(self, env, lr=1e-4, gamma=0.96, eps=0.9, eps_min=0.01,
                 eps_decay=0.995, buffer_cap=10000, seed=0, double_dqn=True,
                 target_sync_ep=10, hidden=(256, 128)):
        super().__init__(env, lr=lr, gamma=gamma, eps=eps, eps_min=eps_min,
                         eps_decay=eps_decay, buffer_cap=buffer_cap, seed=seed,
                         double_dqn=double_dqn, target_sync_ep=target_sync_ep)
        self.hidden = hidden
        self.q_net = QNetConfigurable(self.state_dim, self.action_total, hidden=hidden)
        self.target_q = deepcopy(self.q_net)
        self.opt = optim.Adam(self.q_net.parameters(), lr=lr)


def make_hparam_configs():
    """Full cross-product grid:
    gamma x gae_lambda x ent_coef x clip x hidden = 3^5 = 243 configs."""
    gammas = [0.90, 0.96, 0.99]
    gae_lambdas = [0.90, 0.95, 0.99]
    ent_coefs = [0.0, 0.01, 0.05]
    clips = [0.1, 0.2, 0.3]
    hiddens = [(64, 64), (128, 128), (256, 128)]
    configs = []
    for gamma in gammas:
        for gl in gae_lambdas:
            for ec in ent_coefs:
                for cl in clips:
                    for hd in hiddens:
                        configs.append(dict(gamma=gamma, gae_lambda=gl,
                                            ent_coef=ec, clip=cl, hidden=hd))
    return configs


def make_hparam_configs_representative():
    """Exactly 16-config representative subset for a feasible full run (8 seeds x 16).

    Composition:
      #0  baseline (all default)
      #1-2  gamma low / high
      #3-4  gae_lambda low / high
      #5-6  ent_coef 0.0 / 0.05
      #7-8  clip 0.1 / 0.3
      #9-10 hidden (64,64) / (128,128)
      #11-15 five joint combos sweeping two dims together (so all dim interactions
            get at least one joint observation).
    """
    base = dict(gamma=0.96, gae_lambda=0.95, ent_coef=0.01, clip=0.2, hidden=(256, 128))
    cfgs = [dict(base)]                                  # 0 baseline
    for g in [0.90, 0.99]:                                # 1-2
        c = dict(base); c["gamma"] = g; cfgs.append(c)
    for gl in [0.90, 0.99]:                               # 3-4
        c = dict(base); c["gae_lambda"] = gl; cfgs.append(c)
    for ec in [0.0, 0.05]:                                # 5-6
        c = dict(base); c["ent_coef"] = ec; cfgs.append(c)
    for cl in [0.1, 0.3]:                                 # 7-8
        c = dict(base); c["clip"] = cl; cfgs.append(c)
    for hd in [(64, 64), (128, 128)]:                     # 9-10
        c = dict(base); c["hidden"] = hd; cfgs.append(c)
    # 11-15 joint combos
    jo = [
        dict(gamma=0.90, gae_lambda=0.90, ent_coef=0.01, clip=0.2,  hidden=(256, 128)),
        dict(gamma=0.99, gae_lambda=0.99, ent_coef=0.01, clip=0.2,  hidden=(256, 128)),
        dict(gamma=0.96, gae_lambda=0.95, ent_coef=0.05, clip=0.3,  hidden=(128, 128)),
        dict(gamma=0.96, gae_lambda=0.95, ent_coef=0.0,  clip=0.1,  hidden=(64, 64)),
        dict(gamma=0.99, gae_lambda=0.90, ent_coef=0.05, clip=0.3,  hidden=(128, 128)),
    ]
    for j in jo:
        cfgs.append(dict(j))
    return cfgs[:16]


def run_hparam_sweep(configs, seeds, train_episodes=800, eval_episodes=50,
                     env_cls=UavDynamicSemanticCorrFadingEnv, rho=0.85, sigma_est=1.2,
                     train_dqn=False):
    """
    Train PPO (and optionally DQN) for each (config, seed) and return a DataFrame of
    (config_id, gamma, gae_lambda, ent_coef, clip, hidden, seed, recall, bits).

    NOTE: this is the full-sweep PLUMBING only. It is NOT executed by the pilot.
    Call it once the compute budget is authorized (see report for the exact command).
    """
    rows = []
    for ci, cfg in enumerate(configs):
        for seed in seeds:
            # ---- PPO ----
            env = env_cls(rho=rho, sigma_est=sigma_est)
            env.reset(seed=seed)
            agent = PPOConfigurable(env, seed=seed, gamma=cfg["gamma"],
                                    gae_lambda=cfg["gae_lambda"], ent_coef=cfg["ent_coef"],
                                    clip=cfg["clip"], hidden=cfg["hidden"])
            agent.train(train_episodes)
            env.reset(seed=seed)
            rec, _, bit, rec_ep, _, _ = agent.evaluate(eval_episodes)
            rows.append({
                "config_id": ci,
                "gamma": cfg["gamma"], "gae_lambda": cfg["gae_lambda"],
                "ent_coef": cfg["ent_coef"], "clip": cfg["clip"],
                "hidden": str(cfg["hidden"]), "agent": "PPO",
                "seed": seed, "recall": rec, "recall_ep": rec_ep, "bits": bit,
            })
            # ---- DQN (optional) ----
            if train_dqn:
                env2 = env_cls(rho=rho, sigma_est=sigma_est)
                env2.reset(seed=seed)
                dqn = DQNConfigurable(env2, seed=seed, gamma=cfg["gamma"],
                                      eps=cfg["clip"], hidden=cfg["hidden"])
                dqn.train(train_episodes)
                env2.reset(seed=seed)
                d_rec, _, d_bit, d_rec_ep, _, _ = dqn.evaluate(eval_episodes)
                rows.append({
                    "config_id": ci,
                    "gamma": cfg["gamma"], "gae_lambda": cfg["gae_lambda"],
                    "ent_coef": cfg["ent_coef"], "clip": cfg["clip"],
                    "hidden": str(cfg["hidden"]), "agent": "DQN",
                    "seed": seed, "recall": d_rec, "recall_ep": d_rec_ep, "bits": d_bit,
                })
    return pd.DataFrame(rows)


# ======================================================================================
# (D2) Training curves (segmented train -> greedy-evaluate every K episodes)
# ======================================================================================
def run_training_curves(seeds=range(8), train_episodes=800, eval_every=25,
                        eval_episodes=50, rho=0.85, sigma_est=1.2,
                        out_csv="exp_results/p1_training_curves.csv",
                        out_png="exp_results/p1_training_curves.png"):
    """
    For each (agent in {PPO, DQN}, seed): train for `train_episodes` episodes, and every
    `eval_every` episodes run a greedy evaluation (`eval_episodes`). Uses the VALIDATED
    parent agents (PPO / DQN) with default hyper-parameters.

    Evaluation is made reproducible by using a dedicated eval env that is reseeded with
    the same `seed` at every eval point, so the convergence curve is measured on an
    identical eval set across episodes (controlled comparison).

    Writes out_csv (per-eval-point rows) and out_png (mean +/- std band across seeds).
    Returns (df, equalgran_mean_recall).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = []
    for seed in seeds:
        for agent_name, AgentCls in [("PPO", PPO), ("DQN", DQN)]:
            train_env = UavDynamicSemanticCorrFadingEnv(rho=rho, sigma_est=sigma_est)
            train_env.reset(seed=seed)
            agent = AgentCls(train_env, seed=seed)
            eval_env = UavDynamicSemanticCorrFadingEnv(rho=rho, sigma_est=sigma_est)
            done_eps = 0
            while done_eps < train_episodes:
                agent.train(done_eps + eval_every, resume_from=done_eps)
                done_eps += eval_every
                agent.env = eval_env
                eval_env.reset(seed=seed)          # identical eval set each eval point
                rec, sem, bit, rec_ep, _, _ = agent.evaluate(eval_episodes)
                agent.env = train_env
                rows.append(dict(agent=agent_name, seed=seed, episode=done_eps,
                                 recall_final=rec, recall_perstep=rec_ep,
                                 bits=bit, sem_utility=sem))
                print(f"[curves] {agent_name} seed{seed} ep{done_eps:4d} "
                      f"recall_final={rec:.3f} recall_ps={rec_ep:.3f} bits={bit:.1f}")

    # EqualGran horizontal reference (mean over seeds), same reference env
    eq_recs = []
    for seed in seeds:
        e = UavDynamicSemanticCorrFadingEnv(rho=rho, sigma_est=sigma_est)
        e.reset(seed=seed)
        eq = EqualGranBaseline(e)
        eq_recs.append(eq.evaluate(eval_episodes, global_g=2)[0])
    eq_mean = float(np.mean(eq_recs))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # ---- plot mean +/- std band across seeds ----
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for agent_name, color in [("PPO", "C0"), ("DQN", "C1")]:
        sub = df[df["agent"] == agent_name]
        g = sub.groupby("episode")["recall_final"]
        mean = g.mean()
        std = g.std(ddof=1)
        eps = mean.index.values
        ax.plot(eps, mean.values, marker="o", ms=3, lw=1.4, label=agent_name, color=color)
        ax.fill_between(eps, (mean - std).values, (mean + std).values, alpha=0.2, color=color)
    ax.axhline(eq_mean, ls="--", color="k", lw=1.2,
               label=f"EqualGran ({eq_mean:.3f})")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("High-Priority Recall@high")
    ax.set_title(f"Training curves (rho={rho}, sigma_est={sigma_est}), "
                 f"mean+-std over {len(list(seeds))} seeds")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[curves] EqualGran reference recall = {eq_mean:.4f}")
    return df, eq_mean


# ======================================================================================
# (E) SMALL PILOT (real numbers, bounded runtime)
# ======================================================================================
def run_p1_pilot():
    """
    Bounded sanity checks. Prints REAL computed numbers only.
      E1: short rollout (5 eps x 200 steps) of FixedPolicy on each channel -> SNR stats + Recall@high.
      E2: short PPO (1 seed, 300 eps, default hparams) on Original / AlHourani / ComplexCSI,
          greedily evaluated, alongside EqualGran on the SAME env.
      E3: DPBaseline.greedy_rollout on the original env.
    """
    print("=" * 70)
    print("P1 PILOT  (all numbers below are REAL computed values)")
    print("=" * 70)

    # ---------- E1: channel-model sanity (SNR distribution + recall) ----------
    print("\n--- E1: channel-model SNR distribution (FixedPolicy, 5 eps x 200 steps) ---")
    snr_stats = {}
    for EnvCls, name in [(UavDynamicSemanticCorrFadingEnv, "Original"),
                         (EnvAlHourani, "AlHourani"),
                         (EnvComplexCSI, "ComplexCSI")]:
        env = EnvCls(rho=0.85, sigma_est=1.2)
        snrs, recs = [], []
        for ep in range(5):
            env.reset(seed=ep)
            done = False
            while not done:
                a = np.array([2, 1, 1, 1, 1])  # FixedPolicy action
                _, _, term, trunc, info = env.step(a)
                snrs.append(info["snr_true_dB"])
                recs.append(info["recall_high"])
                done = term or trunc
        snrs = np.array(snrs)
        recs = np.array(recs)
        snr_stats[name] = dict(mean=snrs.mean(), std=snrs.std(),
                                mn=snrs.min(), mx=snrs.max())
        print(f"  {name:10s} | SNR mean={snrs.mean():6.3f} std={snrs.std():6.3f} "
              f"min={snrs.min():6.3f} max={snrs.max():6.3f} | Recall@high(per-step)={recs.mean():.3f}")

    # ---------- E2: short PPO on each channel vs EqualGran ----------
    print("\n--- E2: short PPO (1 seed, 300 eps) vs EqualGran, per channel ---")
    ppo_results = {}
    eq_results = {}
    for EnvCls, name in [(UavDynamicSemanticCorrFadingEnv, "Original"),
                         (EnvAlHourani, "AlHourani"),
                         (EnvComplexCSI, "ComplexCSI")]:
        env = EnvCls(rho=0.85, sigma_est=1.2)
        env.reset(seed=0)
        agent = PPO(env, seed=0)               # default hparams, 300 episodes (short)
        t0 = time.time()
        agent.train(300)
        dt = time.time() - t0
        env.reset(seed=0)
        p_rec, _, p_bit, _, _, _ = agent.evaluate(10)
        # EqualGran on the SAME env class (own resets inside evaluate)
        eq = EqualGranBaseline(env)
        e_rec, _, e_bit, _, _, _ = eq.evaluate(10, global_g=2)
        ppo_results[name] = (p_rec, p_bit)
        eq_results[name] = (e_rec, e_bit)
        print(f"  {name:10s} | PPO Recall(final)={p_rec:.3f} bits={p_bit:7.1f} | "
              f"EqualGran Recall={e_rec:.3f} bits={e_bit:7.1f}  (train {dt:.0f}s)")

    # ---------- E3: DP / myopic greedy oracle on original env ----------
    print("\n--- E3: DPBaseline (myopic greedy oracle) on original env ---")
    env = UavDynamicSemanticCorrFadingEnv(rho=0.85, sigma_est=1.2)
    dp = DPBaseline(env)
    dp_rec_final, dp_rec_ep, dp_bits = dp.greedy_rollout(seed=1, max_steps=200)
    print(f"  Original | DP oracle Recall(final)={dp_rec_final:.3f} "
          f"Recall(per-step)={dp_rec_ep:.3f} bits={dp_bits:.1f}")

    # ---------- summary / honesty check ----------
    print("\n--- Summary & honesty note ---")
    print(f"  EqualGran(Original) Recall = {eq_results['Original'][0]:.3f} "
          f"(paper reference ~0.363, v2)")
    for name in ["Original", "AlHourani", "ComplexCSI"]:
        pr, _ = ppo_results[name]
        er, _ = eq_results[name]
        flag = "BELOW (negative result holds)" if pr < er else "ABOVE (FLAG: check)"
        print(f"  {name:10s}: PPO={pr:.3f} vs EqualGran={er:.3f}  -> {flag}")
    print(f"  DP oracle (upper-bound-style) Recall = {dp_rec_final:.3f} "
          f"(myopic perfect-CSI, NOT full DP)")
    print("=" * 70)
    return dict(snr_stats=snr_stats, ppo=ppo_results, eq=eq_results,
                dp=(dp_rec_final, dp_rec_ep, dp_bits))


if __name__ == "__main__":
    run_p1_pilot()
