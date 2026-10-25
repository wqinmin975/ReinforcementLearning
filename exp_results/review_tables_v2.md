## TABLE IV. Recall@high versus ρ (σ_est = 1.2 dB)

| Method | ρ = 0.5 | ρ = 0.7 | ρ = 0.85 | ρ = 0.95 |
|---|---|---|---|---|
| PPO (policy gradient) | 0.249 ± 0.026 | 0.262 ± 0.046 | 0.252 ± 0.048 | 0.231 ± 0.037 |
| Double-DQN | 0.263 ± 0.032 | 0.290 ± 0.042 | 0.289 ± 0.024 | 0.318 ± 0.028 |
| Fixed policy | 0.221 ± 0.028 | 0.220 ± 0.034 | 0.224 ± 0.033 | 0.229 ± 0.040 |
| EqualGran | 0.360 ± 0.040 | 0.360 ± 0.035 | 0.364 ± 0.033 | 0.368 ± 0.027 |

## TABLE V. Recall@high versus σ_est (in dB) (ρ = 0.85, corrected reward)

| Method | σ_est = 0.0 | σ_est = 0.6 | σ_est = 1.2 | σ_est = 2.0 |
|---|---|---|---|---|
| PPO (policy gradient) | 0.223 ± 0.056 | 0.238 ± 0.039 | 0.252 ± 0.048 | 0.237 ± 0.034 |
| Double-DQN | 0.296 ± 0.049 | 0.305 ± 0.036 | 0.289 ± 0.024 | 0.263 ± 0.030 |
| Fixed policy | 0.224 ± 0.033 | 0.224 ± 0.033 | 0.224 ± 0.033 | 0.224 ± 0.033 |
| EqualGran | 0.364 ± 0.033 | 0.364 ± 0.033 | 0.364 ± 0.033 | 0.364 ± 0.033 |

## TABLE VI. Grid-mean performance summary (16 configurations × 8 seeds, greedy evaluation, corrected reward)

| Method | Recall@high | Semantic utility/step | Bits/step | Params | Latency (ms/step) |
|---|---|---|---|---|---|
| PPO (policy gradient) | 0.241 ± 0.018 | 1.645 ± 0.099 | 110.1 ± 8.5 | 37,395 | 2.329 |
| Double-DQN | 0.289 ± 0.024 | 1.783 ± 0.116 | 120.9 ± 9.3 | 109,248 | 0.138 |
| Fixed policy | 0.223 ± 0.004 | 1.761 ± 0.012 | 143.0 ± 0.0 | 0 | negligible |
| EqualGran | 0.363 ± 0.003 | 3.099 ± 0.015 | 288.0 ± 0.0 | 0 | negligible |

## M7 回合级指标

**M7 回合级总任务价值（基于 6 元组评估）**

| Method | 回合平均 Recall@high | 回合总 Bits/step (mean) |
|---|---|---|
| PPO (policy gradient) | 0.243 | 2884.6 |
| Double-DQN | 0.262 | 2755.9 |
| Fixed policy | 0.234 | 3283.4 |
| EqualGran | 0.313 | 2810.9 |

## 摘要/结论待覆盖数字

[摘要修正后数字] PPO grid-mean Recall@high = 0.241, Double-DQN = 0.289, EqualGran = 0.363; bits/step: PPO = 110.1, DQN = 120.9, EqualGran = 288.0. 学习效率 ≈ 38% (PPO) / 42% (DQN) 的比特成本达到 EqualGran 召回。
