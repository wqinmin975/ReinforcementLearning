# -*- coding: utf-8 -*-
"""
UAV 动态语义通信 — 多随机种子重复实验 / 消融与基线对比
============================================================

依据《多随机种子重复实验.docx》整理而成的完整可运行程序（含后续审查修复）。

研究背景
--------
UAV（无人机）作为空中语义通信中继，需要在有限的通信比特预算与能量预算下，
对地面灾场的图像区域进行「非均匀语义粒度」分配，并同时优化自身飞行航迹，
以最大化高优先级目标（被困人员、坍塌建筑）的语义召回率。

本程序实现：
  * 自定义 Gymnasium 环境 UavDynamicSemanticCorrFadingEnv
        - 每 episode 随机生成 N 个图像区域及其语义重要性（高/中/低优先级）
        - AR(1) 相关瑞利快衰落 + 信道估计噪声（sigma_est）
        - UAV 二维航迹耦合：水平移动影响距离，距离衰减 SNR
        - 联合能耗模型：飞行能耗 + 通信能耗
        - 高优先级区域召回率 Recall@high（带概率模型，非硬判定）
  * 四种对比方案（4.3 消融实验与基线对比）：
        - Proposed PPO : 基于 PPO 联合优化 UAV 航迹 + 非均匀语义粒度
        - DQN baseline : Double-DQN，状态/动作空间与 PPO 完全一致
        - Fixed Policy : 固定 UAV 移动与固定语义粒度（无在线优化）
        - EqualGran    : 所有区域统一粒度，UAV 航迹用朝场景中心的轻量启发式
  * 多随机种子重复实验：每组 (rho, sigma_est) 配置独立重复 N_SEED 次，
    汇报指标 均值 ± 标准差，绘图带误差棒。
  * 复杂度分析：模型参数量 + 单步推理延迟（CPU）。
  * 比特-召回折衷（Pareto）曲线。

【可复现性说明】本程序严格隔离随机源：
  - 环境内部所有随机性统一使用 gym 管理的 self.np_random（由 reset(seed) 播种）；
  - PPO/DQN 各自使用独立的 torch RNG（torch.manual_seed(seed)）与 numpy Generator；
  - 每个 agent 在训练/评估前对所属 env 显式 reset(seed=seed) 一次，后续 episode
    不再重播种，使整段实验由 seed 唯一决定，满足“多种子独立重复”的严谨性要求。

【Checkpoint / 断点续跑】
  - ENABLE_CHECKPOINT=True 时，PPO/DQN 每 save_every 回合落盘模型到
    ./exp_results/checkpoints/，中断后重跑会自动续训（满额则跳过训练直接评估），
    避免数十小时实验从头重来。
  - SELECTED_CONFIGS 形如 [(0.85,1.2)] 可只跑指定 (rho, sigma) 子集；设为 None 跑全部。
  - 注：续跑主要避免重训，续跑后的场景流与“一次性训练”不完全逐位一致；
    若需严格可复现，请一次跑完（或临时设 ENABLE_CHECKPOINT=False）。

依赖
----
    pip install gymnasium numpy matplotlib torch pandas

运行
----
    python uav_semantic_multi_seed_ablation.py
    输出目录 ./exp_results/ 下生成：
        ablation_multi_seed.csv   复杂度外的全部消融结果（均值/标准差）
        complexity.csv            各方案参数量与推理延迟
        ablation_errorbar.png     Recall vs rho / vs sigma_est（带误差棒）
        bit_recall_tradeoff.png   比特-召回折衷曲线
        checkpoints/              PPO/DQN 断点模型（ENABLE_CHECKPOINT=True 时）

提示：完整规模（TRAIN_EPISODES=800, N_SEED=3, 4x4 网格）在 CPU 上耗时较长。
若仅需快速验证，可将 TRAIN_EPISODES、EVAL_EPISODES、N_SEED 调小，或缩小网格 /
用 SELECTED_CONFIGS 只跑部分配置。
"""

import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # 无显示环境下也能保存图片
import matplotlib.pyplot as plt
import pandas as pd

import gymnasium as gym
from gymnasium import spaces


# ====================== 全局参数 ======================
NUM_REGIONS = 12          # 图像区域总数，每 episode 随机生成区域重要性
GRAN_LEVELS = [0, 1, 2, 3]  # 语义粒度等级：0=不传, 1=粗, 2=中, 3=细
bit_cost_table = {0: 0, 1: 8, 2: 24, 3: 64}  # 各粒度对应的传输比特开销

# 语义效用函数 U(g, snr) 与召回概率的奖励/惩罚权重
# B：下调 lambda1 使高优先级区域从 g=1 升到 g=2 的边际净收益转正
#    （bit 代价 16×lambda1 由 0.240 降至 0.128，低于效用增益 ~0.204）
lambda1 = 0.008        # bit 开销惩罚（修订：0.015 → 0.008）
lambda2 = 0.020        # 时延惩罚
lambda_energy = 0.008  # 总能耗惩罚
# A：召回感知奖励项权重，使优化目标与任务指标 Recall@high 对齐
lambda_R = 0.4         # 召回感知奖励（新增）
# 召回项作用域：'high' = 仅高优先级组、等权（与任务指标 (7) Recall@high 同构）；
# 'all' = 原实现（全区域按 s_i 加权，与 (7) 不对齐）。C2 修订后默认 'high'。
RECALL_SCOPE = "high"

# C：高优先级组动作下限（禁止 g<2），通过动作掩码强制高优先级区域至少采用
#    支持召回的粒度；PPO 在策略头施加 logit 掩码，DQN 在 Q 值选择阶段施加掩码。
HIGH_PRIO_MIN_GRAN = 2

# 消融参数网格（论文 4.3）
rho_list = [0.5, 0.7, 0.85, 0.95]                 # 信道时间自相关系数
sigma_est_list = [0.0, 0.6, 1.2, 2.0]             # 信道估计噪声标准差 (dB)
TRAIN_EPISODES = 800   # 每个种子的训练回合数
EVAL_EPISODES = 50     # 每个种子的评估回合数
N_SEED = 3             # 每种参数配置重复的随机种子数（论文常用 3~5，可改为 5）
SAVE_DIR = "./exp_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# 工程开关
ENABLE_CHECKPOINT = True   # 是否启用断点续跑（模型落盘）
SELECTED_CONFIGS = None   # 形如 [(0.85, 1.2)] 仅跑指定 (rho, sigma)；None=全部


def _sstd(x):
    """样本标准差 (ddof=1)，用于 N_SEED 次重复实验的离散度报告（比总体标准差更规范）。
    N_SEED<2 时无法估计离散度，返回 0.0（避免 nan / 告警）。"""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0
    return float(np.std(x, ddof=1))


# ====================== 语义效用 & 召回概率 ======================
def semantic_utility(g, snr_dB):
    """加权语义效用：粒度越高、SNR 越高，效用越大。"""
    if g == 0:
        return 0.0
    base = {1: 0.4, 2: 0.7, 3: 0.95}[g]
    snr_factor = 1.0 / (1 + np.exp(-(snr_dB - 5) / 4))
    return base * snr_factor


def recall_prob(g, snr_dB):
    """关键目标召回概率（仅用于高优先级区域）：粒度 >=2 才有召回可能。"""
    if g <= 1:
        return 0.0
    base_recall = {2: 0.70, 3: 0.92}[g]
    snr_factor = np.clip(snr_dB / 20, 0.0, 1.0)
    return base_recall * snr_factor


# ====================== UAV 环境：AR(1) 相关瑞利衰落 + 信道估计噪声 ======================
class UavDynamicSemanticCorrFadingEnv(gym.Env):
    """
    UAV 动态语义通信环境（含信道时间相关性与估计噪声）。

    状态 (7维, 已归一化到 [0,1])：
        [SNR观测, 可用带宽, 剩余能量, 平均语义重要性, 时延, uav_x, uav_y]
    动作 (MultiDiscrete [4,4,4,3,3])：
        [g_high, g_mid, g_low, dx_code, dy_code]
        - 前三项为 高/中/低 优先级分组的语义粒度 (0~3)
        - dx_code/dy_code 为移动方向编码：0->-1, 1->0, 2->+1

    随机源约定：环境内部所有随机性一律走 self.np_random（由 reset(seed) 播种），
    不触碰模块级全局 np.random，以保证多随机种子实验的可复现性与隔离性。
    """
    metadata = {"render_modes": []}  # 本环境不实现渲染；显式声明避免无意义 render 调用

    def __init__(self, rho=0.85, sigma_est=1.2, render_mode=None, max_episode_steps=200):
        super().__init__()
        self.rho = rho                # 信道时间自相关系数
        self.sigma_est = sigma_est    # 信道估计噪声标准差 (dB)
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps  # 单 episode 步数上限（时间限制截断）
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([4, 4, 4, 3, 3])
        self.delta_inv = {0: -1, 1: 0, 2: 1}   # 动作编码 -> 实际位移
        self.h_prev = None            # AR(1) 信道系数历史
        self.step_count = 0           # 当前 episode 已执行步数

    def _generate_random_regions(self):
        """每 episode 随机生成区域语义重要性 s_i，模拟动态灾害场景。"""
        s_list = []
        for _ in range(NUM_REGIONS):
            r = self.np_random.random()
            if r < 0.25:
                # 高优先级：人、坍塌建筑 s in [0.80, 0.95]
                si = self.np_random.uniform(0.80, 0.95)
            elif r < 0.60:
                # 中优先级：损毁道路 s in [0.50, 0.70]
                si = self.np_random.uniform(0.50, 0.70)
            else:
                # 低优先级：植被空地 s in [0.05, 0.30]
                si = self.np_random.uniform(0.05, 0.30)
            s_list.append(si)
        s_arr = np.array(s_list)
        idx_high = np.where(s_arr >= 0.8)[0]
        idx_mid = np.where((s_arr >= 0.5) & (s_arr < 0.8))[0]
        idx_low = np.where(s_arr < 0.5)[0]
        return s_arr, idx_high, idx_mid, idx_low

    def _sample_correlated_rayleigh(self):
        """AR(1) 相关复高斯信道系数 -> 瑞利衰落功率。"""
        w_real = self.np_random.standard_normal() / np.sqrt(2)
        w_imag = self.np_random.standard_normal() / np.sqrt(2)
        if self.h_prev is None:
            h_real, h_imag = w_real, w_imag
        else:
            h_real = self.rho * self.h_prev[0] + np.sqrt(1 - self.rho ** 2) * w_real
            h_imag = self.rho * self.h_prev[1] + np.sqrt(1 - self.rho ** 2) * w_imag
        self.h_prev = (h_real, h_imag)
        power = h_real ** 2 + h_imag ** 2
        return power

    def _compute_snr_dB(self, distance):
        """路径损耗 (随距离衰减) + 瑞利衰落 -> 真实 SNR(dB)。"""
        snr_path_dB = np.clip(18 - distance * 0.4, 0, 20)
        snr_path_lin = 10 ** (snr_path_dB / 10)
        rayleigh_pow = self._sample_correlated_rayleigh()
        snr_lin = snr_path_lin * rayleigh_pow
        snr_true_dB = 10 * np.log10(max(snr_lin, 1e-8))
        return np.clip(snr_true_dB, 0, 20)

    def _get_obs(self, snr_true):
        """构造观测：在真实 SNR 上叠加估计噪声 sigma_est。"""
        snr_obs = snr_true + self.np_random.normal(0, self.sigma_est)
        snr_obs = np.clip(snr_obs, 0, 20)
        snr_norm = np.clip(snr_obs / 20, 0, 1)
        bw_norm = self.avail_bw
        eng_norm = self.energy_rem / 100
        avg_s_norm = np.mean(self.region_s)
        lat_norm = np.clip(self.latency / 100, 0, 1)
        xu_norm = np.clip(self.xu / 100, -1, 1)
        yu_norm = np.clip(self.yu / 100, -1, 1)
        obs = np.array([snr_norm, bw_norm, eng_norm, avg_s_norm, lat_norm,
                        xu_norm, yu_norm], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.h_prev = None
        self.step_count = 0
        self.region_s, self.idx_high, self.idx_mid, self.idx_low = self._generate_random_regions()
        self.xu = self.np_random.uniform(-30, 30)
        self.yu = self.np_random.uniform(-30, 30)
        d0 = np.sqrt(self.xu ** 2 + self.yu ** 2)
        self.snr_true_dB = self._compute_snr_dB(d0)
        self.avail_bw = self.np_random.uniform(0.2, 1.0)
        self.energy_rem = self.np_random.uniform(30, 100)
        self.latency = 0.0
        observation = self._get_obs(self.snr_true_dB)
        info = {}
        return observation, info

    def step(self, action):
        g_high, g_mid, g_low, dx_code, dy_code = action
        dx = self.delta_inv[dx_code]
        dy = self.delta_inv[dy_code]
        # UAV 位置更新（水平移动）
        self.xu = np.clip(self.xu + dx, -50, 50)
        self.yu = np.clip(self.yu + dy, -50, 50)
        d = np.sqrt(self.xu ** 2 + self.yu ** 2)
        self.snr_true_dB = self._compute_snr_dB(d)

        # 分配粒度：按优先级分组
        g_all = np.zeros(NUM_REGIONS, dtype=int)
        g_all[self.idx_high] = g_high
        g_all[self.idx_mid] = g_mid
        g_all[self.idx_low] = g_low

        # 语义效用、比特开销
        total_sem_utility = 0.0
        total_bits = 0
        recall_sum = 0.0
        num_high = len(self.idx_high)
        for i in range(NUM_REGIONS):
            gi = g_all[i]
            si = self.region_s[i]
            u = semantic_utility(gi, self.snr_true_dB)
            total_sem_utility += si * u
            total_bits += bit_cost_table[gi]
        # 高优先级区域召回率（单独统计）
        if num_high > 0:
            for i in self.idx_high:
                gi = g_all[i]
                p_rec = recall_prob(gi, self.snr_true_dB)
                recall_sum += p_rec
            recall_high = recall_sum / num_high
        else:
            recall_high = 0.0

        # 时延模型
        self.latency = total_bits / (self.avail_bw * 80)

        # 能耗模型：飞行能耗 + 通信能耗
        flight_energy = (abs(dx) + abs(dy)) * 0.8
        comm_energy = total_bits * 0.02
        total_energy_consume = flight_energy + comm_energy
        self.energy_rem = np.clip(self.energy_rem - total_energy_consume, 0, 100)

        # 奖励
        # A：召回感知奖励项。RECALL_SCOPE='high' 时仅对高优先级组等权累加召回概率，
        #    使奖励的召回项与任务指标 (7) Recall@high 在数学上同构（仅差 N_high 常数，
        #    被 λ_R 吸收）；'all' 为旧实现（全区域按 s_i 加权，与 (7) 不对齐）。
        recall_reward = 0.0
        if RECALL_SCOPE == "high":
            for i in self.idx_high:
                recall_reward += recall_prob(g_all[i], self.snr_true_dB)
        else:
            for i in range(NUM_REGIONS):
                recall_reward += self.region_s[i] * recall_prob(g_all[i], self.snr_true_dB)
        reward = (total_sem_utility
                  - lambda1 * total_bits
                  - lambda2 * self.latency
                  - lambda_energy * total_energy_consume
                  + lambda_R * recall_reward)
        # 终止：能量耗尽；截断：达到步数上限。
        # 注意：若不含截断，策略可选"零动作"（不传输+不移动，能耗为 0）使 episode
        # 永不终止，导致 DQN 训练/各方案评估的 while not done 死循环。
        self.step_count += 1
        terminated = (self.energy_rem <= 1e-3)
        truncated = (self.step_count >= self.max_episode_steps) and not terminated
        obs = self._get_obs(self.snr_true_dB)
        info = {
            "sem_utility": total_sem_utility,
            "total_bits": total_bits,
            "recall_high": recall_high,
            "snr_true_dB": self.snr_true_dB,
            "energy_rem": self.energy_rem,
            "uav_pos": (self.xu, self.yu),
        }
        return obs, reward, terminated, truncated, info


# ====================== PPO 模块 ======================
class ActorCriticPPO(nn.Module):
    def __init__(self, state_dim, action_dims, activation="tanh"):
        super().__init__()
        self.action_dims = action_dims
        self.activation = activation
        act_layer = nn.ReLU() if activation == "relu" else nn.Tanh()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256), act_layer,
            nn.Linear(256, 128), act_layer
        )
        self.heads = nn.ModuleList([nn.Linear(128, d) for d in action_dims])
        self.critic = nn.Linear(128, 1)

    def get_action(self, state):
        feat = self.backbone(state)
        acts, logp_sum = [], 0
        for i, h in enumerate(self.heads):
            logits = h(feat)
            # C：高优先级粒度头（index 0）施加动作掩码，禁止 g < HIGH_PRIO_MIN_GRAN
            if i == 0:
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
            # C：与 get_action 一致的掩码，保证训练目标与推理一致
            if i == 0:
                mask_idx = torch.arange(logits.shape[-1]) < HIGH_PRIO_MIN_GRAN
                logits = logits.masked_fill(mask_idx, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            logp_sum += dist.log_prob(actions[:, i])
            entropy_sum += dist.entropy()
        val = self.critic(feat)
        return logp_sum, entropy_sum, val


class PPO:
    def __init__(self, env, lr=3e-4, gamma=0.96, clip=0.2, epoch=10, seed=0,
                 gae_lambda=0.95, activation="tanh"):
        # 仅播种 torch RNG，不污染全局 np.random（环境的随机源已隔离到 self.np_random）
        torch.manual_seed(seed)
        self.seed = seed
        self.env = env
        self.gamma = gamma
        self.clip = clip
        self.epoch = epoch
        self.gae_lambda = gae_lambda
        self.state_dim = env.observation_space.shape[0]
        self.action_dims = env.action_space.nvec.tolist()
        self.ac = ActorCriticPPO(self.state_dim, self.action_dims, activation=activation)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr)

    def count_params(self):
        return sum(p.numel() for p in self.ac.parameters() if p.requires_grad)

    def infer_latency(self, test_runs=200):
        s = torch.randn(1, self.state_dim, dtype=torch.float32)
        t0 = time.time()
        for _ in range(test_runs):
            _ = self.ac.get_action(s)
        t1 = time.time()
        return (t1 - t0) / test_runs

    def save_checkpoint(self, path, episode):
        torch.save({
            "model": self.ac.state_dict(),
            "opt": self.opt.state_dict(),
            "episode": episode,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.ac.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        return int(ckpt.get("episode", 0))

    def collect_traj(self, max_step=200):
        s, _ = self.env.reset()
        traj = {"s": [], "a": [], "r": [], "logp": [], "v": [], "done": [], "info": []}
        for _ in range(max_step):
            st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            a, logp, v = self.ac.get_action(st)
            s_next, r, term, trunc, info = self.env.step(a)
            done = term or trunc
            traj["s"].append(s)
            traj["a"].append(a)
            traj["r"].append(r)
            traj["logp"].append(logp)
            traj["v"].append(v)
            traj["done"].append(done)
            traj["info"].append(info)
            s = s_next
            if done:
                break
        else:
            # for-else：循环因步数上限结束而未触发 done，即“时间限制截断”。
            # 此时末帧并非真正的终止状态，需用 V(s_T) 做 bootstrap；
            # 否则 GAE 会把截断误判为终止（next_val=0），系统性低估回报与优势。
            st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            _, _, v_last = self.ac.get_action(st)
            traj["boot_v"] = float(v_last)
        return traj

    def compute_gae(self, traj):
        rewards, values, dones = traj["r"], traj["v"], traj["done"]
        boot_v = float(traj.get("boot_v", 0.0))  # 时间限制截断时的 bootstrap 值
        gae, adv_list, ret_list = 0, [], []
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # 末帧：真终止 -> 0；时间限制截断 -> V(s_T) bootstrap
                next_val = 0.0 if dones[t] else boot_v
            else:
                next_val = 0.0 if dones[t] else values[t + 1]
            delta = rewards[t] + self.gamma * next_val - values[t]
            gae = delta + self.gamma * self.gae_lambda * (0 if dones[t] else gae)
            adv_list.insert(0, gae)
            ret_list.insert(0, gae + values[t])
        return np.array(adv_list), np.array(ret_list)

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
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return loss.item()

    def train(self, episodes, resume_from=0, ckpt_path=None, save_every=50):
        hist = {"reward": [], "recall_high": [], "sem_util": [], "bit": []}
        for ep in range(resume_from, episodes):
            traj = self.collect_traj()
            self.update(traj)
            ep_r = np.sum(traj["r"])
            # 直接取轨迹最后一帧的 info，避免在已终止状态上再多余地 step 一次
            info = traj["info"][-1]
            hist["reward"].append(ep_r)
            hist["recall_high"].append(info["recall_high"])
            hist["sem_util"].append(info["sem_utility"])
            hist["bit"].append(info["total_bits"])
            if (ep + 1) % 50 == 0:
                print(f"PPO Ep {ep + 1:4d} | R:{ep_r:6.2f}, Recall:{info['recall_high']:.3f}")
            if ckpt_path and (ep + 1) % save_every == 0:
                self.save_checkpoint(ckpt_path, ep + 1)
        if ckpt_path:
            self.save_checkpoint(ckpt_path, episodes)  # 末态存档，标记已完成
        return hist

    def evaluate(self, eval_ep, deterministic=True):
        recalls, sem_utils, bits = [], [], []
        rec_ep, sem_ep, bit_ep = [], [], []
        for _ in range(eval_ep):
            s, _ = self.env.reset()
            done = False
            step_rec, step_sem, step_bit, step_n = 0.0, 0.0, 0.0, 0
            while not done:
                st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                if deterministic:
                    # 评估用确定性（greedy）策略：各头取 argmax，避免采样噪声、
                    # 并消除对 torch RNG 状态的依赖，保证 checkpoint 续跑可复现。
                    # C：高优先级头施加与训练一致的掩码，禁止 g < HIGH_PRIO_MIN_GRAN。
                    head_logits = [h(self.ac.backbone(st)) for h in self.ac.heads]
                    a = np.empty(len(head_logits), dtype=int)
                    for j, hl in enumerate(head_logits):
                        if j == 0:
                            hl = hl.masked_fill(torch.arange(hl.shape[-1]) < HIGH_PRIO_MIN_GRAN, -1e9)
                        a[j] = int(torch.argmax(hl, dim=-1).item())
                else:
                    a, _, _ = self.ac.get_action(st)
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
        return (np.mean(recalls), np.mean(sem_utils), np.mean(bits),
                np.mean(rec_ep), np.mean(sem_ep), np.mean(bit_ep))


# ====================== DQN 模块（Double-DQN） ======================
class QNet(nn.Module):
    def __init__(self, state_dim, action_total):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_total)
        )

    def forward(self, x):
        return self.net(x)


class DQN:
    def __init__(self, env, lr=1e-4, gamma=0.96, eps=0.9, eps_min=0.01,
                 eps_decay=0.995, buffer_cap=10000, seed=0, double_dqn=True,
                 target_sync_ep=10):
        # 仅播种 torch RNG；DQN 自身的探索/采样随机性走独立的 numpy Generator，
        # 不污染环境或全局 np.random 流。
        torch.manual_seed(seed)
        self.seed = seed
        self.rng = np.random.default_rng(seed + 101)
        self.env = env
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.target_sync_ep = target_sync_ep   # target 网络同步周期（episode）
        self.double_dqn = double_dqn           # Double-DQN：online 选动作，target 估值
        self.state_dim = env.observation_space.shape[0]
        self.action_dims = env.action_space.nvec.tolist()
        # 因子化动作的正确编码：扁平下标 = sum_i a_i * stride_i，
        # stride_i = 前 i 个动作维度的连乘（即各分量的权重）。
        # 注意：文档原始实现误用“前缀和”作为偏移，会导致反序列化出越界/负值动作。
        self.strides = [1]
        for d in self.action_dims[:-1]:
            self.strides.append(self.strides[-1] * d)
        self.action_total = int(np.prod(self.action_dims))  # 联合动作空间大小
        # C：高优先级粒度分量（action_dims[0]）在扁平下标中对应 flat_idx % action_dims[0]，
        #    掩码禁止 g_high < HIGH_PRIO_MIN_GRAN（即仅保留 g∈{2,3} 的联合动作）。
        self.valid_mask = torch.tensor(
            [(i % self.action_dims[0]) >= HIGH_PRIO_MIN_GRAN for i in range(self.action_total)],
            dtype=torch.bool)
        self.q_net = QNet(self.state_dim, self.action_total)
        self.target_q = deepcopy(self.q_net)
        self.opt = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = []
        self.buffer_cap = buffer_cap

    def count_params(self):
        return sum(p.numel() for p in self.q_net.parameters() if p.requires_grad)

    def infer_latency(self, test_runs=200):
        s = torch.randn(1, self.state_dim, dtype=torch.float32)
        t0 = time.time()
        for _ in range(test_runs):
            _ = self.q_net(s)
        t1 = time.time()
        return (t1 - t0) / test_runs

    def save_checkpoint(self, path, episode):
        torch.save({
            "model": self.q_net.state_dict(),
            "target": self.target_q.state_dict(),
            "opt": self.opt.state_dict(),
            "eps": self.eps,
            "episode": episode,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.q_net.load_state_dict(ckpt["model"])
        self.target_q.load_state_dict(ckpt["target"])
        self.opt.load_state_dict(ckpt["opt"])
        self.eps = ckpt.get("eps", self.eps_min)
        return int(ckpt.get("episode", 0))

    def _flatten_multi_action(self, act_arr):
        idx = 0
        for i, a in enumerate(act_arr):
            idx += int(a) * self.strides[i]
        return idx

    def _unflatten_action(self, flat_idx):
        rem = int(flat_idx)
        act = []
        # 从最高权重分量向最低权重分量依次解码
        for i in range(len(self.action_dims) - 1, -1, -1):
            val = rem // self.strides[i]
            rem = rem % self.strides[i]
            act.append(val)
        act.reverse()
        return np.array(act, dtype=int)

    def select_action(self, s):
        if self.rng.random() < self.eps:
            # C：随机探索时同样约束高优先级粒度下限
            act = np.array([self.rng.integers(HIGH_PRIO_MIN_GRAN, self.action_dims[0]) if j == 0
                           else self.rng.integers(0, d)
                           for j, d in enumerate(self.action_dims)])
        else:
            st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            q = self.q_net(st)
            # C：掩码掉非法联合动作后取 argmax
            q = q.masked_fill(~self.valid_mask, -1e9)
            flat_idx = torch.argmax(q).item()
            act = self._unflatten_action(flat_idx)
        return act

    def store(self, s, a, r, s_next, done):
        if len(self.buffer) >= self.buffer_cap:
            self.buffer.pop(0)
        self.buffer.append((s, a, r, s_next, done))

    def update(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return
        batch = self.rng.choice(len(self.buffer), batch_size, replace=False)
        s_list, a_list, r_list, sn_list, d_list = [], [], [], [], []
        for bi in batch:
            s, a, r, sn, d = self.buffer[bi]
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            sn_list.append(sn)
            d_list.append(d)
        s = torch.tensor(np.array(s_list), dtype=torch.float32)
        a_flat = torch.tensor([self._flatten_multi_action(aa) for aa in a_list], dtype=torch.int64)
        r = torch.tensor(r_list, dtype=torch.float32)
        sn = torch.tensor(np.array(sn_list), dtype=torch.float32)
        d = torch.tensor(d_list, dtype=torch.float32)
        q_pred = self.q_net(s)
        q_a = q_pred.gather(1, a_flat.unsqueeze(1)).squeeze()
        with torch.no_grad():
            if self.double_dqn:
                # online 网络选动作，target 网络估值，抑制 Q 值过估计
                # C：下一动作的选择同样施加合法动作掩码
                a_next = self.q_net(sn).masked_fill(~self.valid_mask, -1e9).max(1)[1]
                q_next = self.target_q(sn).gather(1, a_next.unsqueeze(1)).squeeze()
            else:
                q_next = self.target_q(sn).max(1)[0]
        target = r + self.gamma * q_next * (1 - d)
        loss = nn.MSELoss()(q_a, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return loss.item()

    def train(self, episodes, resume_from=0, ckpt_path=None, save_every=50, warmup_steps=2000):
        hist = {"reward": [], "recall_high": [], "sem_util": [], "bit": []}
        # 显式预热：先用随机策略填充 replay buffer，避免早期高偏差
        if resume_from == 0 and len(self.buffer) < warmup_steps:
            s, _ = self.env.reset()
            steps = 0
            while steps < warmup_steps:
                # C：随机探索时同样约束高优先级粒度下限
                a = np.array([self.rng.integers(HIGH_PRIO_MIN_GRAN, self.action_dims[0]) if j == 0
                             else self.rng.integers(0, d)
                             for j, d in enumerate(self.action_dims)])
                s_next, r, term, trunc, _ = self.env.step(a)
                done = term or trunc
                self.store(s, a, r, s_next, done)
                s = s_next
                steps += 1
                if done:
                    s, _ = self.env.reset()
        for ep in range(resume_from, episodes):
            s, _ = self.env.reset()
            done = False
            ep_r = 0
            while not done:
                a = self.select_action(s)
                s_next, r, term, trunc, info = self.env.step(a)
                done = term or trunc
                self.store(s, a, r, s_next, done)
                self.update()
                ep_r += r
                s = s_next
            hist["reward"].append(ep_r)
            hist["recall_high"].append(info["recall_high"])
            hist["sem_util"].append(info["sem_utility"])
            hist["bit"].append(info["total_bits"])
            if (ep + 1) % 50 == 0:
                print(f"DQN  Ep {ep + 1:4d} | R:{ep_r:6.2f}, Recall:{info['recall_high']:.3f}")
            # target 网络更频繁同步：每 target_sync_ep 个 episode 硬同步（训练结束再同步一次）
            if (ep + 1) % self.target_sync_ep == 0:
                self.target_q.load_state_dict(self.q_net.state_dict())
            if ckpt_path and (ep + 1) % save_every == 0:
                self.save_checkpoint(ckpt_path, ep + 1)
        if ckpt_path:
            self.save_checkpoint(ckpt_path, episodes)
        return hist

    def evaluate(self, eval_ep):
        recalls, sem_utils, bits = [], [], []
        rec_ep, sem_ep, bit_ep = [], [], []
        for _ in range(eval_ep):
            s, _ = self.env.reset()
            done = False
            step_rec, step_sem, step_bit, step_n = 0.0, 0.0, 0.0, 0
            while not done:
                st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                q = self.q_net(st)
                # C：评估同样施加合法动作掩码
                q = q.masked_fill(~self.valid_mask, -1e9)
                flat_idx = torch.argmax(q).item()
                a = self._unflatten_action(flat_idx)
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
        return (np.mean(recalls), np.mean(sem_utils), np.mean(bits),
                np.mean(rec_ep), np.mean(sem_ep), np.mean(bit_ep))


# ====================== 基线策略：FixedPolicy ======================
class FixedPolicyBaseline:
    """固定 UAV 移动策略与固定语义粒度（无在线优化）。
    固定动作 [2,1,1,1,1]：高/中/低优先级粒度固定，UAV 保持静止（位移编码 1 -> 0）。"""

    def __init__(self, env):
        self.env = env
        self.fixed_action = np.array([2, 1, 1, 1, 1])

    def evaluate(self, eval_ep):
        recalls, sem_utils, bits = [], [], []
        rec_ep, sem_ep, bit_ep = [], [], []
        for _ in range(eval_ep):
            s, _ = self.env.reset()
            done = False
            step_rec, step_sem, step_bit, step_n = 0.0, 0.0, 0.0, 0
            while not done:
                a = self.fixed_action
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
        return (np.mean(recalls), np.mean(sem_utils), np.mean(bits),
                np.mean(rec_ep), np.mean(sem_ep), np.mean(bit_ep))


# ====================== 基线：EqualGran 等粒度（朝中心航迹启发式） ======================
class EqualGranBaseline:
    """等粒度基线：所有区域统一语义粒度 (global_g)，UAV 航迹采用轻量启发式——朝场景
    中心（原点）移动以抬高 SNR。代表“等粒度 + 航迹可优化”但无语义粒度优化的最简对照，
    比随机航迹更公平、也更贴合论文对 EqualGran 的描述。启发式为确定性，不使用随机源。"""

    def __init__(self, env):
        self.env = env

    def _move_toward_center(self):
        # delta_inv: 0->-1, 1->0, 2->+1；朝原点（场景中心）移动，靠近则停。
        dx_code = 1 if abs(self.env.xu) <= 1.0 else (0 if self.env.xu > 0 else 2)
        dy_code = 1 if abs(self.env.yu) <= 1.0 else (0 if self.env.yu > 0 else 2)
        return dx_code, dy_code

    def evaluate(self, eval_ep, global_g=2):
        recalls, sem_utils, bits = [], [], []
        rec_ep, sem_ep, bit_ep = [], [], []
        for _ in range(eval_ep):
            s, _ = self.env.reset()
            done = False
            step_rec, step_sem, step_bit, step_n = 0.0, 0.0, 0.0, 0
            while not done:
                dx_code, dy_code = self._move_toward_center()
                a = np.array([global_g, global_g, global_g, dx_code, dy_code])
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
        return (np.mean(recalls), np.mean(sem_utils), np.mean(bits),
                np.mean(rec_ep), np.mean(sem_ep), np.mean(bit_ep))


# ====================== 批量消融主循环（多种子） ======================
def run_ablation_multi_seed():
    """
    对每一组 (rho, sigma_est) 配置，用 N_SEED 个不同随机种子独立重复实验，
    比较 Proposed PPO / DQN / Fixed Policy / EqualGran 四类方案，
    返回 (结果表 DataFrame, 复杂度表 DataFrame)，并保存 CSV。

    可复现性：每个 agent 创建后立即对所属 env 显式 reset(seed=seed) 一次，
    锁定该 seed 下的随机场景流；后续 episode 的 reset 不带 seed，使整段实验
    由 seed 唯一决定。评估前再次 reset(seed=seed)，使评估场景独立于训练消耗。

    Checkpoint：若 ENABLE_CHECKPOINT 且检测到已有对应 checkpoint，则续训（满额则跳过训练）。
    子集：若 SELECTED_CONFIGS 非 None，仅运行其中包含的 (rho, sigma)。
    """
    results = []
    complexity_records = []
    os.makedirs(SAVE_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(SAVE_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for rho in rho_list:
        for sigma in sigma_est_list:
            if SELECTED_CONFIGS is not None and (rho, sigma) not in SELECTED_CONFIGS:
                print(f"-- skip rho={rho}, sigma={sigma} (not in SELECTED_CONFIGS)")
                continue
            print(f"\n===== rho={rho}, sigma_est={sigma}, run {N_SEED} seeds =====")
            ppo_recall_list, ppo_sem_list, ppo_bit_list = [], [], []
            ppo_rec_ep_list, ppo_bit_ep_list = [], []
            dqn_recall_list, dqn_sem_list, dqn_bit_list = [], [], []
            dqn_rec_ep_list, dqn_bit_ep_list = [], []
            fix_recall_list, fix_sem_list, fix_bit_list = [], [], []
            fix_rec_ep_list, fix_bit_ep_list = [], []
            eq_recall_list, eq_sem_list, eq_bit_list = [], [], []
            eq_rec_ep_list, eq_bit_ep_list = [], []
            for seed in range(N_SEED):
                print(f"--- Seed {seed} ---")
                # PPO
                env = UavDynamicSemanticCorrFadingEnv(rho=rho, sigma_est=sigma)
                env.reset(seed=seed)  # 锁定该种子下的随机场景流（后续 reset 不再传 seed）
                ppo_agent = PPO(env, seed=seed)
                ck = os.path.join(checkpoint_dir, f"ppo_rho{rho}_sig{sigma}_seed{seed}.pt")
                ppo_start = 0
                if ENABLE_CHECKPOINT and os.path.exists(ck):
                    ppo_start = ppo_agent.load_checkpoint(ck)
                    print(f"   [PPO seed{seed}] resume from ep {ppo_start}")
                if ppo_start < TRAIN_EPISODES:
                    ppo_agent.train(TRAIN_EPISODES, resume_from=ppo_start,
                                    ckpt_path=ck if ENABLE_CHECKPOINT else None, save_every=50)
                ppo_agent.env.reset(seed=seed)  # 评估前重播种，使评估场景独立于训练消耗
                p_rec, p_sem, p_bit, p_rec_ep, p_sem_ep, p_bit_ep = ppo_agent.evaluate(EVAL_EPISODES)
                ppo_recall_list.append(p_rec)
                ppo_sem_list.append(p_sem)
                ppo_bit_list.append(p_bit)
                ppo_rec_ep_list.append(p_rec_ep)
                ppo_bit_ep_list.append(p_bit_ep)
                # DQN
                env2 = UavDynamicSemanticCorrFadingEnv(rho=rho, sigma_est=sigma)
                env2.reset(seed=seed)
                dqn_agent = DQN(env2, seed=seed)
                ck2 = os.path.join(checkpoint_dir, f"dqn_rho{rho}_sig{sigma}_seed{seed}.pt")
                dqn_start = 0
                if ENABLE_CHECKPOINT and os.path.exists(ck2):
                    dqn_start = dqn_agent.load_checkpoint(ck2)
                    print(f"   [DQN seed{seed}] resume from ep {dqn_start}")
                if dqn_start < TRAIN_EPISODES:
                    dqn_agent.train(TRAIN_EPISODES, resume_from=dqn_start,
                                    ckpt_path=ck2 if ENABLE_CHECKPOINT else None, save_every=50)
                dqn_agent.env.reset(seed=seed)
                d_rec, d_sem, d_bit, d_rec_ep, d_sem_ep, d_bit_ep = dqn_agent.evaluate(EVAL_EPISODES)
                dqn_recall_list.append(d_rec)
                dqn_sem_list.append(d_sem)
                dqn_bit_list.append(d_bit)
                dqn_rec_ep_list.append(d_rec_ep)
                dqn_bit_ep_list.append(d_bit_ep)
                # Fixed baseline
                env_fix = UavDynamicSemanticCorrFadingEnv(rho=rho, sigma_est=sigma)
                env_fix.reset(seed=seed)
                fix_bl = FixedPolicyBaseline(env_fix)
                f_rec, f_sem, f_bit, f_rec_ep, f_sem_ep, f_bit_ep = fix_bl.evaluate(EVAL_EPISODES)
                fix_recall_list.append(f_rec)
                fix_sem_list.append(f_sem)
                fix_bit_list.append(f_bit)
                fix_rec_ep_list.append(f_rec_ep)
                fix_bit_ep_list.append(f_bit_ep)
                # Equal Gran
                env_eq = UavDynamicSemanticCorrFadingEnv(rho=rho, sigma_est=sigma)
                env_eq.reset(seed=seed)
                eq_bl = EqualGranBaseline(env_eq)
                e_rec, e_sem, e_bit, e_rec_ep, e_sem_ep, e_bit_ep = eq_bl.evaluate(EVAL_EPISODES, global_g=2)
                eq_recall_list.append(e_rec)
                eq_sem_list.append(e_sem)
                eq_bit_list.append(e_bit)
                eq_rec_ep_list.append(e_rec_ep)
                eq_bit_ep_list.append(e_bit_ep)
            # 均值 + 样本标准差 (ddof=1)
            row = {
                "rho": rho, "sigma_est": sigma,
                "PPO_recall_mean": np.mean(ppo_recall_list), "PPO_recall_std": _sstd(ppo_recall_list),
                "PPO_sem_mean": np.mean(ppo_sem_list), "PPO_sem_std": _sstd(ppo_sem_list),
                "PPO_bit_mean": np.mean(ppo_bit_list), "PPO_bit_std": _sstd(ppo_bit_list),
                "PPO_recall_ep_mean": np.mean(ppo_rec_ep_list), "PPO_recall_ep_std": _sstd(ppo_rec_ep_list),
                "PPO_bit_ep_mean": np.mean(ppo_bit_ep_list), "PPO_bit_ep_std": _sstd(ppo_bit_ep_list),
                "DQN_recall_mean": np.mean(dqn_recall_list), "DQN_recall_std": _sstd(dqn_recall_list),
                "DQN_sem_mean": np.mean(dqn_sem_list), "DQN_sem_std": _sstd(dqn_sem_list),
                "DQN_bit_mean": np.mean(dqn_bit_list), "DQN_bit_std": _sstd(dqn_bit_list),
                "DQN_recall_ep_mean": np.mean(dqn_rec_ep_list), "DQN_recall_ep_std": _sstd(dqn_rec_ep_list),
                "DQN_bit_ep_mean": np.mean(dqn_bit_ep_list), "DQN_bit_ep_std": _sstd(dqn_bit_ep_list),
                "Fixed_recall_mean": np.mean(fix_recall_list), "Fixed_recall_std": _sstd(fix_recall_list),
                "Fixed_sem_mean": np.mean(fix_sem_list), "Fixed_sem_std": _sstd(fix_sem_list),
                "Fixed_bit_mean": np.mean(fix_bit_list), "Fixed_bit_std": _sstd(fix_bit_list),
                "Fixed_recall_ep_mean": np.mean(fix_rec_ep_list), "Fixed_recall_ep_std": _sstd(fix_rec_ep_list),
                "Fixed_bit_ep_mean": np.mean(fix_bit_ep_list), "Fixed_bit_ep_std": _sstd(fix_bit_ep_list),
                "EqualGran_recall_mean": np.mean(eq_recall_list), "EqualGran_recall_std": _sstd(eq_recall_list),
                "EqualGran_sem_mean": np.mean(eq_sem_list), "EqualGran_sem_std": _sstd(eq_sem_list),
                "EqualGran_bit_mean": np.mean(eq_bit_list), "EqualGran_bit_std": _sstd(eq_bit_list),
                "EqualGran_recall_ep_mean": np.mean(eq_rec_ep_list), "EqualGran_recall_ep_std": _sstd(eq_rec_ep_list),
                "EqualGran_bit_ep_mean": np.mean(eq_bit_ep_list), "EqualGran_bit_ep_std": _sstd(eq_bit_ep_list),
            }
            results.append(row)
            print(f">>> rho={rho}, sigma={sigma} | PPO Recall {row['PPO_recall_mean']:.3f}±{row['PPO_recall_std']:.3f}")

    # 复杂度（取一组典型环境统计）
    env_comp = UavDynamicSemanticCorrFadingEnv(rho=0.85, sigma_est=1.2)
    ppo_comp = PPO(env_comp, seed=0)
    dqn_comp = DQN(env_comp, seed=0)
    complexity_records.append({"Method": "PPO", "Params": ppo_comp.count_params(),
                               "InferLatency_s": ppo_comp.infer_latency()})
    complexity_records.append({"Method": "DQN", "Params": dqn_comp.count_params(),
                               "InferLatency_s": dqn_comp.infer_latency()})
    # 基线无神经网络，参数量 0，推理时间忽略
    complexity_records.append({"Method": "FixedPolicy", "Params": 0, "InferLatency_s": 0.0})
    complexity_records.append({"Method": "EqualGran", "Params": 0, "InferLatency_s": 0.0})

    df = pd.DataFrame(results)
    df_comp = pd.DataFrame(complexity_records)
    df.to_csv(os.path.join(SAVE_DIR, "ablation_multi_seed.csv"), index=False)
    df_comp.to_csv(os.path.join(SAVE_DIR, "complexity.csv"), index=False)
    return df, df_comp


# ====================== 绘图：消融带误差棒 + 比特召回折衷 ======================
def plot_all_figures(df, fix_sigma=None, fix_rho=None):
    """
    自适应切片（不再硬编码 1.2 / 0.85）：
      - Recall vs rho     : 固定 sigma = fix_sigma（默认取 sigma_est_list 中间值）
      - Recall vs sigma   : 固定 rho   = fix_rho  （默认取 rho_list 中间值）
      - 比特-召回折衷     : 固定 rho   = fix_rho
    使用 np.isclose 做浮点匹配，避免网格调整后切片为空。
    """
    if fix_sigma is None:
        fix_sigma = sigma_est_list[len(sigma_est_list) // 2]
    if fix_rho is None:
        fix_rho = rho_list[len(rho_list) // 2]

    # Fonts sized for print: figures are placed at column/text width in the
    # manuscript, so use physical sizes close to the final print size.
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })

    # ---- shared styling (print-oriented, colour-blind-safe) -----------------
    # Okabe-Ito palette; each scheme keeps the same colour in both figures.
    SCHEMES = [("PPO", "PPO (corrected)", "o"),
               ("DQN", "Double-DQN", "s"),
               ("Fixed", "Fixed policy", "^"),
               ("EqualGran", "EqualGran", "D")]
    PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    EB_KW = dict(markersize=5.0, markeredgecolor="white", markeredgewidth=0.6,
                 linewidth=1.1, elinewidth=1.0, capsize=3, capthick=1.0,
                 alpha=0.92, clip_on=False, zorder=3)

    def _limits_with_err(vals, errs, pad_frac):
        """Axis limits covering every value +/- its error bar, plus margin."""
        v = np.asarray(vals, dtype=float)
        e = np.abs(np.asarray(errs, dtype=float))
        lo, hi = float(np.min(v - e)), float(np.max(v + e))
        pad = pad_frac * max(hi - lo, 1e-9)
        return lo - pad, hi + pad

    # ============ Recall vs rho / vs sigma_est (error-bar panels) ============
    fig1, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))

    def _ablation_panel(ax, sub, xcol, xlabel, title):
        for k, (pre, lab, mk) in enumerate(SCHEMES):
            ax.errorbar(sub[xcol], sub[f"{pre}_recall_mean"],
                        yerr=sub[f"{pre}_recall_std"],
                        marker=mk, color=PALETTE[k], label=lab, **EB_KW)
        # Explicit head-room so error-bar caps never touch / cross the frame.
        means = np.concatenate([sub[f"{p}_recall_mean"].to_numpy() for p, _, _ in SCHEMES])
        errs = np.concatenate([sub[f"{p}_recall_std"].to_numpy() for p, _, _ in SCHEMES])
        ylo, yhi = _limits_with_err(means, errs, pad_frac=0.14)
        ax.set_ylim(ylo, yhi)
        xv = sub[xcol].to_numpy(dtype=float)
        xpad = 0.08 * max(float(xv.max() - xv.min()), 1e-9)
        ax.set_xlim(float(xv.min()) - xpad, float(xv.max()) + xpad)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("High-Priority Recall")
        ax.set_title(title)
        ax.grid(True, alpha=0.30, linewidth=0.6, zorder=0)

    df_s = df[np.isclose(df["sigma_est"], fix_sigma)].sort_values("rho")
    _ablation_panel(axes[0], df_s, "rho", r"$\rho$",
                    r"Recall vs $\rho$ ($\sigma_\mathrm{est}=%s$ dB)" % fix_sigma)
    df_r = df[np.isclose(df["rho"], fix_rho)].sort_values("sigma_est")
    _ablation_panel(axes[1], df_r, "sigma_est", r"$\sigma_\mathrm{est}$ (dB)",
                    r"Recall vs $\sigma_\mathrm{est}$ ($\rho=%s$)" % fix_rho)
    # Shared legend BELOW the axes -> can never occlude data points.
    h1, l1 = axes[0].get_legend_handles_labels()
    fig1.legend(h1, l1, loc="lower center", ncol=4, frameon=False, fontsize=8,
                columnspacing=1.4, handlelength=1.6, bbox_to_anchor=(0.5, 0.02))
    fig1.tight_layout(rect=(0, 0.10, 1, 1))
    fig1.savefig(os.path.join(SAVE_DIR, "ablation_errorbar.png"), dpi=300)
    plt.close(fig1)

    # ============ 比特-召回折衷（Pareto 散点，固定 rho=fix_rho） ==============
    fig2, ax2 = plt.subplots(figsize=(3.35, 3.15))
    df_pareto = df[np.isclose(df["rho"], fix_rho)].sort_values("sigma_est")

    # Lateral dodge (visual only, data untouched): shifts PPO/DQN markers and
    # their near-coincident vertical error bars apart. The shift is bounded by
    # the point's own x error bar, so it never leaves the reported uncertainty;
    # zero-error baselines (Fixed/EqualGran) are therefore not moved.
    x_all = np.concatenate([df_pareto[f"{p}_bit_mean"].to_numpy() for p, _, _ in SCHEMES])
    e_all = np.concatenate([df_pareto[f"{p}_bit_std"].to_numpy() for p, _, _ in SCHEMES])
    xlo2, xhi2 = _limits_with_err(x_all, e_all, pad_frac=0.10)
    dodge_step = 0.05 * (xhi2 - xlo2)
    DODGE = {"PPO": -1.0, "DQN": +1.0, "Fixed": 0.0, "EqualGran": 0.0}

    for k, (pre, lab, mk) in enumerate(SCHEMES):
        xb = df_pareto[f"{pre}_bit_mean"].to_numpy(dtype=float)
        yb = df_pareto[f"{pre}_recall_mean"].to_numpy(dtype=float)
        ex = np.abs(df_pareto[f"{pre}_bit_std"].to_numpy(dtype=float))
        ey = np.abs(df_pareto[f"{pre}_recall_std"].to_numpy(dtype=float))
        # sigma-invariant baselines produce identical stacked markers; keep 1st
        keep, seen = [], set()
        for i in range(len(xb)):
            key = (round(float(xb[i]), 6), round(float(yb[i]), 6))
            if key not in seen:
                seen.add(key)
                keep.append(i)
        xb, yb, ex, ey = xb[keep], yb[keep], ex[keep], ey[keep]
        xoff = DODGE[pre] * np.minimum(dodge_step, 0.5 * ex)
        ax2.errorbar(xb + xoff, yb, xerr=ex, yerr=ey, marker=mk,
                     color=PALETTE[k], label=lab, **EB_KW)

    y_all = np.concatenate([df_pareto[f"{p}_recall_mean"].to_numpy() for p, _, _ in SCHEMES])
    ye_all = np.concatenate([df_pareto[f"{p}_recall_std"].to_numpy() for p, _, _ in SCHEMES])
    ylo2, yhi2 = _limits_with_err(y_all, ye_all, pad_frac=0.14)
    ax2.set_ylim(ylo2, yhi2)
    ax2.set_xlim(xlo2, xhi2)
    ax2.set_xlabel("Total Transmitted Bits")
    ax2.set_ylabel("High-Priority Recall")
    ax2.set_title("Bit-Recall Trade-off Curve")
    ax2.grid(True, alpha=0.30, linewidth=0.6, zorder=0)
    h2, l2 = ax2.get_legend_handles_labels()
    fig2.legend(h2, l2, loc="lower center", ncol=2, frameon=False, fontsize=7.5,
                columnspacing=1.0, handlelength=1.5, bbox_to_anchor=(0.5, 0.01))
    fig2.tight_layout(rect=(0, 0.17, 1, 1))

    # ---- inset zoom on the dense 100-165 bits cluster ----------------------
    # Created AFTER tight_layout (tight_layout would otherwise treat the inset
    # as a regular axes and collapse the main panel). Placed in the empty
    # lower-right of the axes (EqualGran sits separately at ~288 bits / high
    # recall), so it never occludes any data point.
    axin = ax2.inset_axes([0.50, 0.06, 0.47, 0.46])
    for k, (pre, lab, mk) in enumerate(SCHEMES):
        xi = df_pareto[f"{pre}_bit_mean"].to_numpy(dtype=float)
        yi = df_pareto[f"{pre}_recall_mean"].to_numpy(dtype=float)
        ei = np.abs(df_pareto[f"{pre}_bit_std"].to_numpy(dtype=float))
        eyi = np.abs(df_pareto[f"{pre}_recall_std"].to_numpy(dtype=float))
        kp, sv = [], set()
        for i in range(len(xi)):
            key = (round(float(xi[i]), 6), round(float(yi[i]), 6))
            if key not in sv:
                sv.add(key); kp.append(i)
        xi, yi, ei, eyi = xi[kp], yi[kp], ei[kp], eyi[kp]
        xoffi = DODGE[pre] * np.minimum(dodge_step, 0.5 * ei)
        # clip_on=True here (overrides EB_KW): points outside the zoom window
        # (e.g. PPO sigma=0, EqualGran) must not bleed outside the inset.
        axin.errorbar(xi + xoffi, yi, xerr=ei, yerr=eyi, marker=mk,
                      color=PALETTE[k], **{**EB_KW, "clip_on": True})
    axin.set_xlim(100, 165)
    axin.set_ylim(0.190, 0.345)
    axin.set_xlabel("Bits", fontsize=7)
    axin.set_ylabel("Recall", fontsize=7)
    axin.tick_params(labelsize=6.5)
    axin.grid(True, alpha=0.30, linewidth=0.5, zorder=0)
    ax2.indicate_inset_zoom(axin, edgecolor="0.45", alpha=0.7, linewidth=0.9)
    fig2.savefig(os.path.join(SAVE_DIR, "bit_recall_tradeoff.png"), dpi=300)
    plt.close(fig2)


if __name__ == "__main__":
    df_result, df_complex = run_ablation_multi_seed()
    plot_all_figures(df_result)
    print("\n===== Complexity Table =====")
    print(df_complex)
