# UAV Semantic Communication — Reinforcement Learning Core

This repository contains the **core simulation and reinforcement-learning (RL) code** for the
study of UAV-assisted semantic communication with adaptive, bit-budget-constrained control.

The work trains lightweight RL agents (**PPO** and **Double-DQN**) to jointly steer the UAV
trajectory and choose the semantic-compression granularity under a strict **per-step transmit-bit
budget**, and benchmarks them against fixed-granularity (`EqualGran`) and stationary-UAV baselines.
The central finding is that the learned agents achieve competitive semantic recall at roughly
40% lower bit cost than the equal-granularity baseline, rather than superior raw recall.

> Corresponding paper: under review at *Computer Networks* and *Physical Communication*
> (Elsevier). If you use this code, please cite the published version once available.

## What's inside

- `uav_semantic_multi_seed_ablation.py` — **central module**: the semantic-communication
  environment, the PPO / Double-DQN agents, the `make_env` factory, the greedy-policy
  evaluators, and the multi-seed ablation driver. Most other scripts import from it.
- `p1_variants.py`, `run_p1_full.py`, `run_p1_5d.py`, `run_p1_5e.py` — the P1 channel-coefficient
  study (variance of the semantic-channel coefficient) and its run scripts.
- `run_full_scale.py` — full-scale multi-seed experiment entry point.
- `run_decomp.py`, `run_levene.py`, `run_budget_win.py`, `run_persistep_means.py`,
  `run_review_experiments.py` — factorial decomposition, Levene variance tests, budget-win-region,
  per-step means, and reviewer-requested experiment runners.
- `aerpaw_ds19_cir.py`, `aerpaw_ds24_full.py`, `aerpaw_validate.py` — validation against AERPAW
  over-the-air channel sounding data (out-of-lab channel applicability check).
- `p1_empirical_run.py`, `p1_empirical_probe.py`, `p1_coeff_sensitivity.py` — empirical P1 probing
  and sensitivity analysis.
- `measure_bit_profile.py`, `measure_ep_len.py` — per-step bit-profile and episode-length
  measurement utilities (import the central module).
- `capture_limits.py`, `diag_ppo.py`, `smoke_test_reward.py`, `probe_figdata.py`,
  `make_p1_coeff_figure.py`, `regen_figs.py` — supporting diagnostics and figure regeneration.

## Results data

`exp_results/` contains the CSV summaries and PNG figures used in the paper: ablation error bars,
the bit–recall trade-off (with inset zoom), the empirical P1 coefficient, training curves, and the
decomposition / Levene / budget-win tables.

> Note: trained model checkpoints (`exp_results/checkpoints/*.pt`, ~0.5 GB) and the raw 22 MB
> per-seed `bit_profile_perseed.csv` are **not** included — they are large binary/derived
> artifacts and are fully reproducible by running the scripts above.

## Requirements

See `requirements.txt`. Developed and tested with Python 3.13.

```
torch>=1.12
numpy>=1.21
pandas>=1.3
matplotlib>=3.5
scipy>=1.7
```

## Quick start

Run the main multi-seed ablation (writes CSV / PNG into `exp_results/`):

```bash
python run_full_scale.py            # or: python uav_semantic_multi_seed_ablation.py
```

Run the P1 channel-coefficient study:

```bash
python run_p1_full.py
```

Validate against AERPAW channel sounding:

```bash
python aerpaw_validate.py
```

## License

Code is provided for research purposes. Please cite the associated paper.
