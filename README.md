# CartPole HCRL + RLHF

> **Capstone Project** · Master's in Statistical Machine Learning
> Hanoi University of Science and Technology
> Instructor: Assoc. Prof. Thân Quang Khoát

A CartPole-v1 balancing agent using **tabular Q-Learning** extended with two human-feedback paradigms:

- **HCRL** (Human-Centered RL / TAMER) — human gives real-time scalar feedback (+/−) at individual timesteps; a learned `HCRLRewardModel` fills in predictions when the human is silent
- **RLHF** (Reinforcement Learning from Human Preferences, Christiano et al. 2017) — human compares pairs of trajectory clips; a learned `RewardModel` drives the agent

Every method comes in **two flavours**:

| Script | Mode | Oracle / Human |
|---|---|---|
| `train_hcrl.py` | HCRL automated | Simulated oracle (`oracle_feedback`) |
| `train_hcrl_human.py` | HCRL interactive | Real human (arrow keys) |
| `train_rlhf.py` | RLHF automated | Simulated oracle (`oracle_preference`) |
| `train_rlhf_human.py` | RLHF interactive | Real human (pygame clip comparison) |

The project also investigates:

1. **Timing** — when during training is HCRL feedback most effective (early / mid / late / throughout)?
2. **Magnitude** — how does the scale of the feedback signal affect learning?

---

## Table of Contents

- [Background](#background)
- [Method](#method)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start — Full Command Sequence](#quick-start--full-command-sequence)
- [Scripts Reference](#scripts-reference)
  - [run.py — Baseline](#runpy--baseline)
  - [train_hcrl.py — HCRL (oracle)](#train_hcrlpy--hcrl-oracle)
  - [train_hcrl_human.py — HCRL (human)](#train_hcrl_humanpy--hcrl-human)
  - [train_rlhf.py — RLHF (oracle)](#train_rlhfpy--rlhf-oracle)
  - [train_rlhf_human.py — RLHF (human)](#train_rlhf_humanpy--rlhf-human)
  - [feedback_timing_experiment.py](#feedback_timing_experimentpy)
  - [sensitivity_analysis.py](#sensitivity_analysispy)
  - [run_all.py — Full automated pipeline](#run_allpy--full-automated-pipeline)
  - [compare_all.py — Big-picture comparison](#compare_allpy--big-picture-comparison)
  - [compare_models.py](#compare_modelspy)
  - [convergence_analysis.py](#convergence_analysispy)
  - [analyze_feedback.py](#analyze_feedbackpy)
  - [watch.py](#watchpy)
  - [visual_compare.py](#visual_comparepy)
  - [webapp.py — Web Visualizer](#webapppy--web-visualizer)
- [Output Directory Layout](#output-directory-layout)
- [Hyperparameter Reference](#hyperparameter-reference)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Background

### CartPole-v1

A pole is mounted on a cart moving along a frictionless track. The agent pushes left or right to keep the pole upright. Episode ends when:

- Pole angle exceeds ±12° (±0.2095 rad)
- Cart position exceeds ±2.4 units
- 200 timesteps elapsed (success)

**Solved** = mean episode length ≥ 195 over 30 consecutive episodes.

### TAMER (Knox & Stone, 2009)

Human trainer gives real-time scalar feedback (+/−) that shapes the agent's reward signal. Key claims:
- Human feedback accelerates convergence beyond environment reward alone
- Feedback is most useful **later in training**, when the agent has a semi-stable policy to correct
- Feedback scale must be calibrated relative to the environment reward

### RLHF (Christiano et al., 2017)

Instead of per-timestep labels, the human watches **pairs of trajectory clips** and says which looks better. A neural reward model learns from these pairwise preferences, then drives Q-learning — the agent never sees the true environment reward after warm-up.

---

## Method

### Q-Learning Agent

| Parameter | Value |
|---|---|
| State features | Cart position, velocity; pole angle, angular velocity |
| Discretization bins | 7 per feature → 4,096 total states |
| Actions | 2 (push left / push right) |
| Learning rate α | 0.05 (all methods — fair comparison) |
| Discount factor γ | 0.95 (all methods — fair comparison) |
| Initial exploration ε₀ | 0.50 |
| Exploration decay | 0.99 per episode |

Q-update rule:
```
Q(s,a) ← Q(s,a) + α · [r_total + γ · max Q(s',a') − Q(s,a)]
```

### HCRL Reward Signal

At each timestep the agent receives a shaped reward:

```
shaped = oracle_signal        (if oracle / human fired)
shaped = reward_model(obs)    (if silent and model trained)
shaped = env_reward           (fallback: model not yet trained)

r_total = shaped − TERMINATE_PENALTY   (on early termination)
r_total = shaped                       (normal step)
```

### HCRL Reward Model (`HCRLRewardModel`)

Trained by **MSE regression** on `(observation, feedback)` pairs accumulated across all episodes so far. Retrained after every episode. This lets the agent receive a meaningful reward signal even when the oracle/human is silent.

```
Loss = (1/N) Σ (r̂(sᵢ) − hᵢ)²     hᵢ ∈ {+WEIGHT, −WEIGHT}
```

### RLHF Reward Model (`RewardModel`)

Trained by **cross-entropy on pairwise preferences** (Bradley-Terry model):

```
P̂(A ≻ B) = exp(Σ r̂(sₜᴬ)) / (exp(Σ r̂(sₜᴬ)) + exp(Σ r̂(sₜᴮ)))
Loss     = −[ μ · log P̂(A≻B) + (1−μ) · log P̂(B≻A) ]
```

`μ = 1` if human (or oracle) preferred clip A, `0` if B, `0.5` if tie.

Both reward models share architecture: **2-layer MLP** (obs_dim → 64 → 64 → 1, tanh activations) trained with Adam. Saved as `.npz`.

### Oracle (Automated Human)

For fully reproducible experiments:

- **HCRL oracle** (`oracle_feedback`) — gives ±WEIGHT feedback based on pole angle and cart position stability. Fires with 50% probability per timestep (models human reaction time).
- **RLHF oracle** (`oracle_preference`) — Boltzmann-rational model that picks the better clip with probability proportional to `exp(sum_of_rewards / T)`.

---

## Project Structure

```
ML-Project/
│
├── cartpole/                          # Core library
│   ├── __init__.py                    # Public API exports
│   ├── agents.py                      # Agent (ABC), RandomActionAgent, QLearningAgent
│   ├── entities.py                    # EpisodeHistory, EpisodeHistoryRecord, type aliases
│   ├── plotting.py                    # Live Matplotlib training plot
│   ├── oracle.py                      # Simulated human oracles (HCRL + RLHF)
│   └── reward_model.py                # RewardModel (RLHF) + HCRLRewardModel (TAMER)
│
├── run.py                             # Baseline Q-Learning, 3 seeds
│
├── train_hcrl.py                      # HCRL oracle (automated, simulated human)
├── train_hcrl_human.py                # HCRL human (interactive, arrow keys)
├── train_rlhf.py                      # RLHF oracle (automated, simulated human)
├── train_rlhf_human.py                # RLHF human (interactive, pygame clip viewer)
│
├── feedback_timing_experiment.py      # Timing: Early/Mid/Late/Full × 3 seeds
├── sensitivity_analysis.py            # Weight sensitivity: [5,20,50] × 3 seeds
├── run_all.py                         # Full automated pipeline (no human scripts)
│
├── compare_all.py                     # Big-picture: all 12 methods, one chart
├── compare_models.py                  # Training curves + gameplay for HCRL conditions
├── convergence_analysis.py            # Threshold-crossing convergence speed
├── analyze_feedback.py                # Analyze human feedback logs
│
├── watch.py                           # Watch a single model play (pygame)
├── visual_compare.py                  # Up to 6 models side-by-side (pygame)
├── webapp.py                          # Flask web visualizer (stream to browser)
│
├── tests/
│   ├── test_episode_history.py
│   ├── test_qlearning_agent.py
│   └── test_random_agent.py
│
└── experiment-results/                # All outputs (models, CSVs, plots)
```

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

**Install uv:**
```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -Ls https://astral.sh/uv/install.sh | sh
```

**Install dependencies:**
```bash
uv sync
```

---

## Quick Start — Full Command Sequence

Run all steps in this order (replace `100` with your desired episode count).  
Steps 1–6 are **fully automated** (no human required). Steps 7–8 require you at the keyboard.

```bash
# ── Step 1: Baseline ─────────────────────────────────────────────────────────
uv run python run.py --episodes 100

# ── Step 2: HCRL oracle (automated, oracle_feedback + HCRLRewardModel) ───────
uv run python train_hcrl.py --episodes 100 --seed 0

# ── Step 3: RLHF oracle (automated, oracle_preference + RewardModel) ─────────
uv run python train_rlhf.py --episodes 100 --seed 0

# ── Step 4: HCRL timing experiment (4 conditions × 3 seeds) ──────────────────
uv run python feedback_timing_experiment.py --auto --episodes 100

# ── Step 5: HCRL sensitivity analysis (3 weights × 3 seeds) ──────────────────
uv run python sensitivity_analysis.py --episodes 100

# ── Step 6: Compare all automated methods ────────────────────────────────────
uv run python compare_all.py --episodes 100 --seed 0

# ── Step 7 (optional, interactive): HCRL with real human arrow-key feedback ──
uv run python train_hcrl_human.py --episodes 100 --seed 0

# ── Step 8 (optional, interactive): RLHF with real human clip comparisons ────
uv run python train_rlhf_human.py --episodes 100 --seed 0

# ── Step 9 (optional): Re-run compare_all after interactive steps ─────────────
uv run python compare_all.py --episodes 100 --seed 0
```

**One-command automated pipeline** (steps 1–6 combined, skips interactive scripts):
```bash
uv run python run_all.py --episodes 100
```

---

## Scripts Reference

### `run.py` — Baseline

Pure Q-Learning with no human feedback. Trains 3 seeds.

```bash
uv run python run.py
uv run python run.py --episodes 200
uv run python run.py --episodes 500 --verbose
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--episodes` | int | 100 | Training episodes per seed |
| `--verbose` | flag | off | Render window + live matplotlib plot |

**Output:** `experiment-results/ep{N}/`
- `baseline_s{0,1,2}_model.npz`
- `baseline_s{0,1,2}_history.csv`

---

### `train_hcrl.py` — HCRL (oracle)

Automated HCRL using a simulated oracle. No human or display needed. Identical agent hyperparameters to the baseline for fair comparison.

**Pipeline per step:**
1. `oracle_feedback()` fires with 50% probability → adds to reward model buffer
2. Oracle silent + model trained → `HCRLRewardModel.predict(obs)`
3. Oracle silent + model not yet trained → fallback to env reward (+1/step)
4. End of episode → retrain `HCRLRewardModel` on all collected signals

```bash
uv run python train_hcrl.py --episodes 100 --seed 0
uv run python train_hcrl.py --episodes 200 --seed 1
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--episodes` | int | 100 | Total training episodes |
| `--seed` | int | 0 | Random seed |

**Key constants:**

| Constant | Default | Description |
|---|---|---|
| `ORACLE_TRIGGER_PROB` | 0.5 | Probability oracle fires each timestep |
| `FEEDBACK_WEIGHT` | 10.0 | Magnitude of +/− oracle signal |
| `TERMINATE_PENALTY` | 50.0 | Penalty on early episode termination |
| `REWARD_MODEL_EPOCHS` | 20 | Gradient steps per episode retraining |
| `REWARD_MODEL_LR` | 1e-3 | Adam LR for HCRLRewardModel |

**Output:** `experiment-results/ep{N}/hcrl-oracle/`
- `hcrl_oracle_s{seed}_model.npz`
- `hcrl_oracle_s{seed}_reward_model.npz`
- `hcrl_oracle_s{seed}_history.csv`
- `hcrl_oracle_s{seed}_results.png`

---

### `train_hcrl_human.py` — HCRL (human)

Interactive HCRL: CartPole renders live while you press arrow keys to give real-time feedback. `HCRLRewardModel` fills in predictions when you are silent.

**Pipeline per step:**
1. Human presses `↑` or `↓` → oracle signal collected; reward model buffer updated
2. Silent + model trained → `HCRLRewardModel.predict(obs)`
3. Silent + model not yet trained → fallback to env reward
4. End of episode → retrain `HCRLRewardModel` on all collected signals

> **Tip:** Click the CartPole window once after it opens so it captures your keystrokes.

```bash
uv run python train_hcrl_human.py --episodes 100 --seed 0
uv run python train_hcrl_human.py --episodes 200 --seed 1
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--episodes` | int | 100 | Total training episodes |
| `--seed` | int | 0 | Random seed |

**Controls:**

| Key | Effect |
|---|---|
| `↑` Arrow Up | Positive feedback (+10) — good move! |
| `↓` Arrow Down | Negative feedback (−10) — bad move! |
| `Esc` | Quit early (saves progress) |

**Key constants:**

| Constant | Default | Description |
|---|---|---|
| `FEEDBACK_WEIGHT` | 10.0 | Magnitude of +/− keypress signal |
| `TERMINATE_PENALTY` | 50.0 | Penalty on early episode termination |
| `STEP_DELAY` | 0.05 s | Seconds per step (slows for reaction time) |
| `REWARD_MODEL_EPOCHS` | 20 | Gradient steps per episode retraining |
| `REWARD_MODEL_LR` | 1e-3 | Adam LR for HCRLRewardModel |

**Output:** `experiment-results/ep{N}/hcrl-human/`
- `hcrl_human_s{seed}_model.npz`
- `hcrl_human_s{seed}_reward_model.npz`
- `hcrl_human_s{seed}_history.csv`
- `hcrl_human_s{seed}_results.png`

---

### `train_rlhf.py` — RLHF (oracle)

Automated RLHF using a simulated oracle (no human needed). Agent hyperparameters are identical to the baseline for fair comparison.

**Pipeline:**
1. Warm-up (20% of episodes): train agent on env reward; collect trajectory segments
2. Bootstrap: sample segment pairs → oracle labels → train `RewardModel`
3. RLHF loop: run episodes with reward model signal → collect segments → human labels → retrain

```bash
uv run python train_rlhf.py --episodes 100 --seed 0
uv run python train_rlhf.py --episodes 200 --seed 1
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--episodes` | int | 100 | Total training episodes |
| `--seed` | int | 0 | Random seed |

**Key constants:**

| Constant | Default | Description |
|---|---|---|
| `WARMUP_FRACTION` | 0.20 | Fraction of episodes used as env-reward warm-up |
| `EPISODES_PER_ITER` | 8 | Policy episodes per RLHF iteration |
| `SEGMENT_LENGTH` | 25 | Timesteps per trajectory clip |
| `WARMUP_SEGMENTS` | 40 | Initial clips before RL starts |
| `SEGMENTS_PER_ITER` | 8 | New clips per iteration |
| `PAIRS_PER_ITER` | 24 | Preference queries per reward-model update |
| `REWARD_MODEL_EPOCHS` | 40 | Gradient steps per update |
| `SEGMENT_BUFFER_SIZE` | 400 | Max clips in replay buffer |
| `REWARD_MODEL_LR` | 3e-4 | Adam LR for RewardModel |

**Output:** `experiment-results/ep{N}/rlhf-oracle/`
- `rlhf_oracle_s{seed}_model.npz`
- `rlhf_oracle_s{seed}_reward_model.npz`
- `rlhf_oracle_s{seed}_history.csv`
- `rlhf_oracle_s{seed}_results.png`

---

### `train_rlhf_human.py` — RLHF (human)

Interactive RLHF: a pygame window shows pairs of CartPole clips. You press `A`/`B`/`S` to label which looks better. Clip length and FPS scale up as training progresses (short/slow early → longer/faster later).

```bash
uv run python train_rlhf_human.py --episodes 100 --seed 0
uv run python train_rlhf_human.py --episodes 200 --seed 1
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--episodes` | int | 100 | Total training episodes |
| `--seed` | int | 0 | Random seed |

**Controls during labelling:**

| Key | Meaning |
|---|---|
| `A` | Clip A was better (μ = 1.0) |
| `B` | Clip B was better (μ = 0.0) |
| `S` | Skip / tie (μ = 0.5) |
| `Esc` | Stop early, save and exit |

**Key constants:**

| Constant | Default | Description |
|---|---|---|
| `WARMUP_FRACTION` | 0.20 | Fraction of episodes used as env-reward warm-up |
| `EPISODES_PER_ITER` | 8 | Policy episodes per RLHF iteration |
| `SEGMENT_LENGTH` | 50 | Starting clip length (scales to 100 by last iteration) |
| `CLIP_FPS` | 30 | Starting playback FPS (scales to 45 by last iteration) |
| `WARMUP_PAIRS` | 10 | Pairs shown during bootstrap round |
| `PAIRS_PER_ITER` | 4 | Pairs labelled per iteration (low to avoid fatigue) |
| `REWARD_MODEL_EPOCHS` | 50 | Gradient steps after each labelling round |
| `REWARD_MODEL_LR` | 3e-4 | Adam LR for RewardModel |

**Output:** `experiment-results/ep{N}/rlhf-human/`
- `rlhf_human_s{seed}_model.npz`
- `rlhf_human_s{seed}_reward_model.npz`
- `rlhf_human_s{seed}_history.csv`
- `rlhf_human_s{seed}_results.png`

---

### `feedback_timing_experiment.py`

Tests **when** during training HCRL feedback is most effective. Trains 4 conditions × 3 seeds = 12 runs. Each condition uses `oracle_feedback` + `HCRLRewardModel`.

| Condition | Feedback window |
|---|---|
| Early | 0% → 20% of episodes |
| Mid | 40% → 60% of episodes |
| Late | 80% → 100% of episodes |
| Full Feedback | 0% → 100% (always on) |

Outside the feedback window, the `HCRLRewardModel` (trained on in-window data) continues to predict rewards. This tests whether feedback learned early/mid/late generalizes.

```bash
# Automated (oracle + HCRLRewardModel, no keyboard)
uv run python feedback_timing_experiment.py --auto --episodes 100
uv run python feedback_timing_experiment.py --auto --episodes 200

# Analyze only (re-generate charts from saved results)
uv run python feedback_timing_experiment.py --analyze --episodes 100

# Interactive (you give feedback via keyboard — seed 0 only)
uv run python feedback_timing_experiment.py --episodes 100
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--episodes` | int | 100 | Training episodes per condition |
| `--auto` | flag | off | Use oracle instead of keyboard |
| `--analyze` | flag | off | Skip training, run analysis only |
| `--skip-charts` | flag | off | Skip `plt.show()` |

**Key constants:**

| Constant | Default | Description |
|---|---|---|
| `SEEDS` | [0, 1, 2] | Random seeds |
| `MAX_TIMESTEPS` | 200 | Episode length cap |
| `TERMINATE_PENALTY` | 5000 | Early-termination penalty |
| `feedback_weight` | 10.0 | Oracle reward magnitude |

**Output:** `experiment-results/ep{N}/timing-experiment/`
- `{condition}_s{seed}_model.npz`
- `{condition}_s{seed}_reward_model.npz`
- `{condition}_s{seed}_history.csv`
- `{condition}_s{seed}_feedback_log.csv`
- `timing_experiment_results.png`

Analysis prints a **Mann-Whitney U test** (one-sided) for each condition vs. baseline.

---

### `sensitivity_analysis.py`

Tests how **feedback magnitude** affects learning. Full Feedback window (all episodes), 3 seeds × 3 weights = 9 runs. Each run uses `oracle_feedback` + `HCRLRewardModel`.

```bash
uv run python sensitivity_analysis.py --episodes 100
uv run python sensitivity_analysis.py --episodes 200

# Analyze only
uv run python sensitivity_analysis.py --episodes 100 --analyze
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--episodes` | int | 100 | Training episodes per weight |
| `--analyze` | flag | off | Skip training, run analysis only |
| `--skip-charts` | flag | off | Skip `plt.show()` |

**Key constants:**

| Constant | Default | Description |
|---|---|---|
| `FEEDBACK_WEIGHTS` | [5, 20, 50] | Reward magnitudes to test |
| `SEEDS` | [0, 1, 2] | Random seeds |
| `MAX_TIMESTEPS` | 200 | Episode length cap |
| `TERMINATE_PENALTY` | 5000 | Early-termination penalty |

**Output:** `experiment-results/ep{N}/sensitivity/`
- `w{weight}_s{seed}_model.npz`
- `w{weight}_s{seed}_reward_model.npz`
- `w{weight}_s{seed}.csv`
- `sensitivity_results.png`

---

### `run_all.py` — Full Automated Pipeline

Runs the complete automated experiment in one command:  
baseline → timing experiment → sensitivity analysis → compare models → convergence analysis.

No human input required.

```bash
uv run python run_all.py --episodes 100
uv run python run_all.py --episodes 200

# Skip training, re-run analysis only
uv run python run_all.py --episodes 200 --analyze-only

# Headless (no plt.show)
uv run python run_all.py --episodes 200 --skip-charts
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--episodes` | int | 200 | Training episodes per condition |
| `--eval-episodes` | int | 100 | Gameplay evaluation episodes per model |
| `--analyze-only` | flag | off | Skip all training |
| `--skip-charts` | flag | off | Skip chart steps |

---

### `compare_all.py` — Big-picture comparison

Compares **all 12 methods** under the same conditions (same `--episodes`, same `--seed`). Missing models are skipped gracefully.

**Methods compared:**

| Group | Methods |
|---|---|
| Baseline | Baseline Q-Learning |
| HCRL | HCRL Early, Mid, Late, Full (timing experiment) |
| HCRL | HCRL oracle (`train_hcrl.py`) |
| HCRL | HCRL human (`train_hcrl_human.py`) |
| Sensitivity | HCRL w=5, w=20, w=50 |
| RLHF | RLHF oracle, RLHF human |

```bash
uv run python compare_all.py --episodes 100 --seed 0
uv run python compare_all.py --episodes 200 --seed 0 --eval-episodes 200
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--episodes` | int | 100 | Training episode count for all methods |
| `--seed` | int | 0 | Which seed to load for each method |
| `--eval-episodes` | int | 100 | Gameplay evaluation episodes per model |

**Output:** `experiment-results/`
- `compare_all_training.png` — all learning curves overlaid + paradigm-group means
- `compare_all_gameplay.png` — box plot, bar chart, histogram
- Console: full stats table + Welch t-test + Cohen's d vs baseline for every method

---

### `compare_models.py`

Compare Baseline + 4 HCRL timing conditions. Multi-seed aware (mean ± std shaded bands). Runs Welch's t-test + Cohen's d for each HCRL model vs. baseline.

```bash
uv run python compare_models.py --episodes 100
uv run python compare_models.py --episodes 200
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--episodes` | int | 100 | Which episode-count results to load |
| `--eval-episodes` | int | 100 | Evaluation gameplay episodes per model |

**Output:** `experiment-results/ep{N}/comparison_training.png`, `comparison_gameplay.png`

---

### `convergence_analysis.py`

Measures how fast each model first crosses episode-length thresholds of **50, 100, 150, 200** using a rolling mean.

```bash
uv run python convergence_analysis.py --episodes 100
uv run python convergence_analysis.py --episodes 200
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--episodes` | int | 100 | Which episode-count results to load |

**Output:** `experiment-results/ep{N}/convergence_analysis.png`. Also prints AUC per model — higher = better training efficiency.

---

### `analyze_feedback.py`

Analyze timing, frequency, and state distribution of feedback from interactive HCRL sessions.

```bash
uv run python analyze_feedback.py experiment-results/ep100
uv run python analyze_feedback.py --compare experiment-results/ep100/timing-experiment
```

---

### `watch.py`

Watch a single trained model play in a pygame window.

```bash
uv run python watch.py experiment-results/ep100/hcrl-oracle/hcrl_oracle_s0_model.npz
uv run python watch.py experiment-results/ep100/rlhf-oracle/rlhf_oracle_s0_model.npz 20
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `model_path` | path | required | Path to `.npz` agent file |
| `num_episodes` | int | 10 | Episodes to watch |

---

### `visual_compare.py`

Watch up to **6 models play simultaneously** in a dynamic pygame grid.

| Models | Grid |
|---|---|
| 1–2 | 1 × 2 |
| 3–4 | 2 × 2 |
| 5–6 | 2 × 3 |

```bash
# Compare oracle vs human methods (2×2 grid)
uv run python visual_compare.py \
  experiment-results/ep100/ep100/baseline_s0_model.npz \
  experiment-results/ep100/hcrl-oracle/hcrl_oracle_s0_model.npz \
  experiment-results/ep100/hcrl-human/hcrl_human_s0_model.npz \
  experiment-results/ep100/rlhf-oracle/rlhf_oracle_s0_model.npz \
  --labels Baseline "HCRL oracle" "HCRL human" "RLHF oracle" \
  --episodes 10
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `models` | path(s) | required | 1–6 paths to `.npz` files |
| `--labels` | str(s) | filename stem | Display labels |
| `--episodes` | int | 10 | Episodes to play |

---

### `webapp.py` — Web Visualizer

Watch trained models in a browser — no pygame required.

```bash
uv run python webapp.py
# Open: http://localhost:5000
```

Features:
- Auto-discovers all agent `.npz` models from `experiment-results/`, grouped by type
- Episode slider (1–30) and speed slider (5–60 fps)
- Live frame streaming via Server-Sent Events (SSE)
- Per-model stats: current steps, running mean, best episode

**Expose publicly with ngrok:**
```bash
# Terminal 1
uv run python webapp.py

# Terminal 2
ngrok http 5000
```

---

## Output Directory Layout

```
experiment-results/
│
└── ep{N}/                                  # All results for N-episode runs
    │
    ├── baseline_s{0,1,2}_model.npz         # run.py
    ├── baseline_s{0,1,2}_history.csv
    │
    ├── hcrl-oracle/                         # train_hcrl.py
    │   ├── hcrl_oracle_s{seed}_model.npz
    │   ├── hcrl_oracle_s{seed}_reward_model.npz
    │   ├── hcrl_oracle_s{seed}_history.csv
    │   └── hcrl_oracle_s{seed}_results.png
    │
    ├── hcrl-human/                          # train_hcrl_human.py
    │   ├── hcrl_human_s{seed}_model.npz
    │   ├── hcrl_human_s{seed}_reward_model.npz
    │   ├── hcrl_human_s{seed}_history.csv
    │   └── hcrl_human_s{seed}_results.png
    │
    ├── rlhf-oracle/                         # train_rlhf.py
    │   ├── rlhf_oracle_s{seed}_model.npz
    │   ├── rlhf_oracle_s{seed}_reward_model.npz
    │   ├── rlhf_oracle_s{seed}_history.csv
    │   └── rlhf_oracle_s{seed}_results.png
    │
    ├── rlhf-human/                          # train_rlhf_human.py
    │   ├── rlhf_human_s{seed}_model.npz
    │   ├── rlhf_human_s{seed}_reward_model.npz
    │   ├── rlhf_human_s{seed}_history.csv
    │   └── rlhf_human_s{seed}_results.png
    │
    ├── timing-experiment/                   # feedback_timing_experiment.py
    │   ├── {early,mid,late,full_feedback}_s{seed}_model.npz
    │   ├── {early,mid,late,full_feedback}_s{seed}_reward_model.npz
    │   ├── {early,mid,late,full_feedback}_s{seed}_history.csv
    │   ├── {early,mid,late,full_feedback}_s{seed}_feedback_log.csv
    │   └── timing_experiment_results.png
    │
    ├── sensitivity/                         # sensitivity_analysis.py
    │   ├── w{5,20,50}_s{seed}_model.npz
    │   ├── w{5,20,50}_s{seed}_reward_model.npz
    │   ├── w{5,20,50}_s{seed}.csv
    │   └── sensitivity_results.png
    │
    ├── comparison_training.png              # compare_models.py
    ├── comparison_gameplay.png
    └── convergence_analysis.png            # convergence_analysis.py
```

---

## Hyperparameter Reference

### Agent (all methods — identical for fair comparison)

| Parameter | Value | Notes |
|---|---|---|
| Learning rate α | 0.05 | Consistent across all scripts |
| Discount factor γ | 0.95 | Consistent across all scripts |
| Initial exploration ε₀ | 0.50 | 50% random at start |
| Exploration decay | 0.99/ep | ~13% at ep 200 |
| State bins | 7/feature | 4,096 total states |

### HCRL Oracle / Human

| Parameter | Value | Notes |
|---|---|---|
| Oracle trigger probability | 0.50 | Models human reaction time |
| Feedback weight (oracle/human) | 10.0 | Magnitude of +/− signal |
| Terminate penalty | 50.0 | Proportional to feedback scale |
| Reward model LR | 1e-3 | Adam, HCRLRewardModel |
| Reward model hidden | 64 | 4→64→64→1, tanh |
| Reward model epochs/ep | 20 | Retrained after every episode |

### Timing Experiment / Sensitivity Analysis

| Parameter | Value | Notes |
|---|---|---|
| Feedback weights (sensitivity) | 5, 20, 50 | — |
| Seeds | [0, 1, 2] | 3 seeds for statistical validity |
| Terminate penalty | 5000 | Historical value kept for comparability |

### RLHF Reward Model

| Parameter | Value | Notes |
|---|---|---|
| Architecture | MLP 4→64→64→1 | tanh activations |
| Optimiser | Adam | β₁=0.9, β₂=0.999 |
| Learning rate | 3e-4 | RewardModel |
| Loss | Cross-entropy (Bradley-Terry) | Pairwise preferences |
| Segment length (oracle) | 25 timesteps | Scales up during human training |
| Segment length (human) | 50 → 100 timesteps | Grows with iteration number |

---

## Development

```bash
uv sync                     # install all deps including dev tools

uv run ruff check .         # lint
uv run ty check             # type check
uv run pytest               # tests
```

Pre-commit hooks run automatically on `git commit` after:
```bash
uv run pre-commit install
```

---

## Troubleshooting

**Matplotlib/Tk window not showing on Linux:**
```bash
sudo apt install python3-tk
```

**UnicodeEncodeError on Windows:**
```bash
set PYTHONIOENCODING=utf-8
```
All scripts call `sys.stdout.reconfigure(encoding="utf-8")` at startup.

**`ngrok: command not found`:**
Add ngrok to PATH or use full path: `C:\path\to\ngrok.exe http 5000`.

**Model file not found in `visual_compare` / `watch`:**
Paths are relative to the current working directory. Always run from the project root.

**Pygame window freezes during interactive training:**
Click the window once to bring it into focus before pressing keys.

---

## References

Christiano, P., Leike, J., Brown, T. B., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

Knox, W. B., & Stone, P. (2009). Interactively shaping agents via human reinforcement: The TAMER framework. *Proceedings of the 5th International Conference on Knowledge Capture (K-CAP)*, pp. 9–16. ACM. https://doi.org/10.1145/1597735.1597738

Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983). Neuronlike adaptive elements that can solve difficult learning control problems. *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-13(5), 834–846.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
