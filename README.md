# CartPole HCRL + RLHF

> **Capstone Project** · Master's in Statistical Machine Learning
> Hanoi University of Science and Technology
> Instructor: Assoc. Prof. Thân Quang Khoát

Implementation and comparison of two human-feedback RL paradigms from the research papers:

| Paper | Algorithm | Training Script |
|-------|-----------|-----------------|
| Knox & Stone (2009); Li et al. (2019) | **TAMER / HCRL** — per-timestep scalar feedback | `train_hcrl.py` |
| Knox & Stone (2012); Li et al. (2019) | **VI-TAMER** — non-myopic TAMER with value function | `train_vi_tamer.py` |
| Christiano et al. (2017) | **RLHF** — pairwise trajectory preferences | `train_rlhf.py` |
| Christiano et al. (2017) §2.2 full | **RLHF + Ensemble** — ensemble, uncertainty queries, reward normalisation | `train_rlhf_ensemble.py` |

---

## Table of Contents

- [Background](#background)
- [Algorithm Overview](#algorithm-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Scripts Reference](#scripts-reference)
- [Hyperparameter Reference](#hyperparameter-reference)
- [Output Directory Layout](#output-directory-layout)
- [Development](#development)
- [References](#references)

---

## Background

### CartPole-v1

A pole is mounted on a cart. The agent pushes left or right to keep the pole upright.
Episode terminates when:
- Pole angle > ±12° (±0.2095 rad)
- Cart position > ±2.4 units
- 200 timesteps elapsed (= success)

**Solved** = mean episode length ≥ 195 over 30 consecutive episodes.

### TAMER — Knox & Stone (2009)

Human trainer gives real-time scalar feedback (+/−) at individual timesteps.
A learned `HCRLRewardModel` (MLP regression) generalises those signals to
all states, filling silence gaps between oracle firings.

Key research questions tested in this project:
- **Timing**: when during training is feedback most effective (early / mid / late)?
- **Magnitude**: how does feedback scale affect learning speed?

### VI-TAMER — Knox & Stone (2012)

Extends TAMER from myopic (γ=0) to non-myopic (γ>0) by adding a value
function `Q_H(s,a)` driven by the human reward model `R̂_H`:

```
TAMER    (γ=0):  policy = argmax_a R̂_H(s,a)          [immediate only]
VI-TAMER (γ>0):  Q_H(s,a) ← Q_H + α·(R̂_H + γ·max Q_H(s') − Q_H)
                 policy    = argmax_a Q_H(s,a)          [discounted future]
```

Setting γ=0 exactly recovers plain TAMER.

### RLHF — Christiano et al. (2017)

Human watches **pairs of trajectory clips** and indicates which looks better.
A neural `RewardModel` (Bradley-Terry / cross-entropy) learns from these
pairwise preferences and drives the RL agent — no environment reward needed
after warm-up.

Full §2.2 improvements (in `train_rlhf_ensemble.py`):
1. **Ensemble** (K=3 models, bootstrapped training) — §2.2 bullet 1
2. **Uncertainty-based query selection** (highest ensemble variance) — §2.2.4
3. **Reward normalisation** (zero mean / unit std, Welford online) — §2.2.1
4. **Oracle human-error noise** (10% uniform random responses) — §2.2.3

---

## Algorithm Overview

### Q-Learning Agent (shared across all methods)

| Parameter | Value | Configured in |
|-----------|-------|---------------|
| State discretization | 7 bins/feature → 4,096 states | `config.NUM_BINS` |
| Learning rate α | 0.05 | `config.AGENT_LR` |
| Discount factor γ | 0.95 | `config.AGENT_DISCOUNT` |
| Initial exploration ε₀ | 0.50 | `config.AGENT_EXPLORE` |
| Exploration decay | 0.99/episode | `config.AGENT_DECAY` |

### Reward signals per method

```
Baseline   → env_reward (+1/step) or −BASELINE_TERMINATE_PENALTY

HCRL/TAMER → oracle_signal          (if oracle fired, 50% prob)
           → reward_model.predict() (if silent and model trained)
           → env_reward             (fallback before first training)

VI-TAMER   → same as HCRL for oracle signal
           → Q_H updated via: R̂_H(s,a) + γ·max Q_H(s')  [TD non-myopic]

RLHF       → reward_model.predict(obs)  (trained on pairwise preferences)
           → env_reward during warm-up phase
```

### Reward model architectures (both share `_MLPBase`)

```
Architecture:  Linear(4, 64) → tanh → Linear(64, 64) → tanh → Linear(64, 1)
Optimizer:     Adam (β₁=0.9, β₂=0.999, ε=1e-8)

HCRLRewardModel: loss = MSE(r̂(s), h)          [scalar regression, HCRL]
RewardModel:     loss = cross-entropy(μ, P̂)   [preference model, RLHF]
EnsembleRewardModel: K × RewardModel, bootstrapped, with uncertainty scoring
```

---

## Project Structure

```
ML-Project/
│
├── cartpole/                       # Core library (Python package)
│   ├── __init__.py                 # Public exports
│   ├── config.py                   # ALL hyperparameters — single source of truth
│   ├── agents.py                   # QLearningAgent, VITAMERAgent, RandomActionAgent
│   │                               # Shared _DiscretizationMixin base class
│   ├── entities.py                 # EpisodeHistory, type aliases
│   ├── oracle.py                   # oracle_feedback() — HCRL simulated oracle
│   ├── reward_model.py             # _MLPBase, RewardModel, HCRLRewardModel,
│   │                               # EnsembleRewardModel, oracle_preference()
│   ├── train_utils.py              # Shared training helpers used by all scripts:
│   │                               #   run_hcrl_episode, run_vi_tamer_episode,
│   │                               #   run_rl_episode, collect_segment,
│   │                               #   sample_preference_pairs, evaluate_agent,
│   │                               #   make_agent, save_history_csv, ...
│   └── plotting.py                 # Live Matplotlib training plot
│
├── run.py                          # Baseline Q-Learning (3 seeds)
│
├── train_hcrl.py                   # HCRL/TAMER — simulated oracle (automated)
├── train_hcrl_human.py             # HCRL — real human (arrow keys, interactive)
├── train_vi_tamer.py               # VI-TAMER — non-myopic, simulated oracle
├── train_rlhf.py                   # RLHF baseline — single reward model
├── train_rlhf_ensemble.py          # RLHF full §2.2 — ensemble + all improvements
├── train_rlhf_human.py             # RLHF — real human (pygame clip comparison)
│
├── feedback_timing_experiment.py   # When is feedback most effective? (4 conditions)
├── sensitivity_analysis.py         # Feedback magnitude study (weights 5/20/50)
├── run_all.py                      # Full automated pipeline orchestrator
│
├── compare_all.py                  # Big-picture comparison of all methods
├── compare_models.py               # Training curves + gameplay for HCRL conditions
├── compare_rlhf.py                 # RLHF oracle vs human comparison
├── convergence_analysis.py         # Threshold-crossing convergence speed
├── analyze_feedback.py             # Analyze feedback patterns from logs
│
├── replay.py                       # Replay a saved agent model
├── watch.py                        # Watch a model play (pygame)
├── visual_compare.py               # Up to 6 models side-by-side (pygame)
├── webapp.py                       # Flask web visualizer (stream to browser)
│
├── paper/
│   ├── R6-Human-centered reinforcement learning- A survey -19.pdf
│   └── R6-Deep reinforcement learning from human preferences -17.pdf
│
├── tests/
│   ├── test_episode_history.py
│   ├── test_qlearning_agent.py
│   └── test_random_agent.py
│
└── experiment-results/             # All outputs (auto-created)
    └── ep{N}/
        ├── baseline_s{seed}_model.npz
        ├── baseline_s{seed}_history.csv
        ├── hcrl-oracle/
        ├── vi-tamer/
        ├── rlhf-oracle/
        ├── rlhf-ensemble/
        ├── timing-experiment/
        └── sensitivity/
```

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (Windows PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install uv (macOS / Linux)
curl -Ls https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

---

## Quick Start

All automated steps — no human interaction required.

```bash
# 1. Baseline Q-Learning (control condition)
uv run python run.py --episodes 200

# 2. HCRL / TAMER  (Li et al. 2019, Knox & Stone 2009)
uv run python train_hcrl.py --episodes 200 --seed 0

# 3. VI-TAMER  (Knox & Stone 2012, non-myopic)
uv run python train_vi_tamer.py --episodes 200 --seed 0
uv run python train_vi_tamer.py --episodes 200 --seed 0 --gamma 0   # γ=0 → plain TAMER

# 4. RLHF baseline  (Christiano et al. 2017)
uv run python train_rlhf.py --episodes 200 --seed 0

# 5. RLHF + Ensemble  (Christiano et al. 2017, full §2.2)
uv run python train_rlhf_ensemble.py --episodes 200 --seed 0

# 6. Timing experiment  (when is feedback most effective?)
uv run python feedback_timing_experiment.py --auto --episodes 200

# 7. Sensitivity analysis  (how does feedback magnitude matter?)
uv run python sensitivity_analysis.py --episodes 200

# 8. Full pipeline (runs steps 1, 6, 7, then analysis)
uv run python run_all.py --episodes 200
```

Interactive training (real human feedback):

```bash
# HCRL: press Arrow Up (good) / Arrow Down (bad) while watching
uv run python train_hcrl_human.py

# RLHF: watch two clips, click to choose the better one
uv run python train_rlhf_human.py
```

---

## Scripts Reference

### Training Scripts

| Script | Paper | Notes |
|--------|-------|-------|
| `run.py` | — | Baseline Q-Learning, 3 seeds |
| `train_hcrl.py` | Li et al. (2019) / Knox & Stone (2009) | TAMER, oracle |
| `train_hcrl_human.py` | Li et al. (2019) / Knox & Stone (2009) | TAMER, real human |
| `train_vi_tamer.py` | Knox & Stone (2012) | VI-TAMER, `--gamma` controls γ |
| `train_rlhf.py` | Christiano et al. (2017) | Single RewardModel |
| `train_rlhf_ensemble.py` | Christiano et al. (2017) §2.2 | Ensemble + all improvements |
| `train_rlhf_human.py` | Christiano et al. (2017) | Real human clip comparison |

### Experiment Scripts

| Script | Research Question |
|--------|------------------|
| `feedback_timing_experiment.py` | When is feedback most effective? (Early 0-20% / Mid 40-60% / Late 80-100% / Full) |
| `sensitivity_analysis.py` | How does feedback magnitude affect learning? (weights 5/20/50) |
| `run_all.py` | Full automated pipeline orchestrator |

### Analysis Scripts

| Script | Output |
|--------|--------|
| `compare_all.py` | Learning curves + gameplay for all 10+ methods |
| `compare_models.py` | Pairwise training curve + gameplay comparison |
| `compare_rlhf.py` | RLHF oracle vs human comparison |
| `convergence_analysis.py` | Episodes-to-threshold convergence speed |
| `analyze_feedback.py` | Feedback density, timing, magnitude patterns |

### Visualization Scripts

| Script | Usage |
|--------|-------|
| `replay.py` | `uv run python replay.py experiment-results/ep200/hcrl-oracle/hcrl_oracle_s0_model.npz` |
| `watch.py` | `uv run python watch.py <model.npz>` |
| `visual_compare.py` | `uv run python visual_compare.py <model1.npz> <model2.npz> ...` (up to 6) |
| `webapp.py` | `uv run python webapp.py` → open `http://localhost:5000` |

---

## Hyperparameter Reference

All constants live in **`cartpole/config.py`** — one file to change them all.

### Agent (shared across all methods)

| Constant | Value | Description |
|----------|-------|-------------|
| `AGENT_LR` | 0.05 | Q-table learning rate α |
| `AGENT_DISCOUNT` | 0.95 | Discount factor γ |
| `AGENT_EXPLORE` | 0.50 | Initial ε-greedy exploration |
| `AGENT_DECAY` | 0.99 | Per-episode exploration decay |
| `NUM_BINS` | 7 | Discretization bins per state feature |
| `MAX_TIMESTEPS` | 200 | CartPole episode cap |
| `GOAL_LENGTH` | 195 | Mean episode length for success |
| `GOAL_CONSECUTIVE` | 30 | Consecutive episodes for success |

### HCRL / TAMER  (Knox & Stone 2009; Li et al. 2019)

| Constant | Value | Description |
|----------|-------|-------------|
| `HCRL_TRIGGER_PROB` | 0.5 | Oracle firing probability per timestep |
| `HCRL_FEEDBACK_WEIGHT` | 10.0 | ±magnitude of oracle signal |
| `HCRL_TERMINATE_PENALTY` | 50.0 | Penalty on early episode termination |
| `HCRL_RM_LR` | 1e-3 | HCRLRewardModel Adam learning rate |
| `HCRL_RM_HIDDEN` | 64 | Hidden layer width |
| `HCRL_RM_EPOCHS` | 20 | Gradient steps per reward-model update |

### VI-TAMER  (Knox & Stone 2012)

| Constant | Value | Description |
|----------|-------|-------------|
| `VI_TAMER_DISCOUNT` | 0.95 | Default γ for Q_H; use `--gamma 0` for plain TAMER |

### RLHF  (Christiano et al. 2017)

| Constant | Value | Description |
|----------|-------|-------------|
| `RLHF_SEGMENT_LENGTH` | 25 | Timesteps per trajectory clip |
| `RLHF_WARMUP_FRACTION` | 0.20 | Fraction of episodes for warm-up |
| `RLHF_EPISODES_PER_ITER` | 8 | Policy episodes per RLHF iteration |
| `RLHF_WARMUP_SEGMENTS` | 40 | Segments collected during warm-up |
| `RLHF_SEGMENTS_PER_ITER` | 8 | New segments per iteration |
| `RLHF_PAIRS_PER_ITER` | 24 | Preference queries per reward-model update |
| `RLHF_RM_EPOCHS` | 40 | Gradient steps per reward-model update |
| `RLHF_SEGMENT_BUFFER` | 400 | Rolling segment replay buffer size |
| `RLHF_RM_LR` | 3e-4 | RewardModel Adam learning rate |
| `RLHF_RM_HIDDEN` | 64 | Hidden layer width |

### RLHF §2.2 Ensemble improvements

| Constant | Value | Description |
|----------|-------|-------------|
| `ENSEMBLE_N_MODELS` | 3 | Number of ensemble members K |
| `ENSEMBLE_CANDIDATES_MULT` | 10 | Candidate pairs = mult × PAIRS_PER_ITER |
| `ENSEMBLE_ERROR_PROB` | 0.1 | Oracle human-error rate (§2.2.3) |
| `ENSEMBLE_VAL_FRACTION` | ≈1/e | Validation split for each bootstrapped model |

### Experiment design

| Constant | Value | Description |
|----------|-------|-------------|
| `SEEDS` | [0, 1, 2] | Random seeds for statistical validity |
| `SENSITIVITY_WEIGHTS` | [5, 20, 50] | Feedback magnitudes tested |
| `TIMING_CONDITIONS` | early/mid/late/full | Feedback window fractions |

---

## Output Directory Layout

```
experiment-results/
└── ep200/                          # Results for --episodes 200
    ├── baseline_s0_model.npz       # Q-table weights (seed 0)
    ├── baseline_s0_history.csv     # Episode lengths (seed 0)
    ├── baseline_model.npz          # Canonical seed-0 copy
    ├── episode_history.csv         # Canonical seed-0 history
    │
    ├── hcrl-oracle/                # HCRL with simulated oracle
    │   ├── hcrl_oracle_s0_model.npz
    │   ├── hcrl_oracle_s0_reward_model.npz
    │   └── hcrl_oracle_s0_history.csv
    │
    ├── vi-tamer/                   # VI-TAMER (non-myopic TAMER)
    │   ├── vi_tamer_s0_model.npz
    │   └── vi_tamer_s0_history.csv
    │
    ├── rlhf-oracle/                # RLHF single reward model
    │   ├── rlhf_oracle_s0_model.npz
    │   ├── rlhf_oracle_s0_reward_model.npz
    │   └── rlhf_oracle_s0_history.csv
    │
    ├── rlhf-ensemble/              # RLHF with ensemble + §2.2 improvements
    │   ├── rlhf_ensemble_s0_model.npz
    │   ├── rlhf_ensemble_s0_model_0.npz  (ensemble member 0)
    │   ├── rlhf_ensemble_s0_model_1.npz
    │   ├── rlhf_ensemble_s0_model_2.npz
    │   ├── rlhf_ensemble_s0_normalizer.npz
    │   └── rlhf_ensemble_s0_history.csv
    │
    ├── timing-experiment/          # 4 conditions × 3 seeds
    │   ├── early_s0_model.npz
    │   ├── early_s0_history.csv
    │   ├── early_s0_feedback_log.csv
    │   └── ... (mid, late, full_feedback)
    │
    └── sensitivity/                # Weights [5, 20, 50] × 3 seeds
        ├── w5_s0_model.npz
        ├── w5_s0_history.csv
        └── ...
```

---

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check .

# Format
uv run black .
uv run isort .
```

---

## References

| Paper | Key contribution |
|-------|-----------------|
| Knox & Stone (2009). *Interactively Shaping Agents via Human Reinforcement: The TAMER Framework.* K-CAP. | TAMER algorithm: per-timestep scalar human feedback; credit assignment; HCRLRewardModel |
| Knox & Stone (2012). *Reinforcement Learning from Human Reward and Advice.* AAMAS. | VI-TAMER: non-myopic value function Q_H driven by human reward model |
| Li et al. (2019). *Human-Centered Reinforcement Learning: A Survey.* IEEE THMS. | Survey of HCRL algorithms; categorizes interactive shaping, categorical feedback, policy feedback |
| Christiano et al. (2017). *Deep Reinforcement Learning from Human Preferences.* NeurIPS. | RLHF: pairwise trajectory preferences; Bradley-Terry reward model; ensemble + uncertainty queries |
