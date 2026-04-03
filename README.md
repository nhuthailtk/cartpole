# CartPole HCRL + RLHF

> **Capstone Project** · Master's in Statistical Machine Learning
> Hanoi University of Science and Technology
> Instructor: Assoc. Prof. Thân Quang Khoát

A CartPole-v1 balancing agent using **tabular Q-Learning** extended with two human-feedback paradigms:

- **HCRL** (Human-Centered RL / TAMER) — human gives real-time scalar feedback (+/−) at individual timesteps
- **RLHF** (Reinforcement Learning from Human Preferences, Christiano et al. 2017) — human compares pairs of trajectory clips; a learned reward model drives the agent

The project investigates two research questions under HCRL:

1. **Timing** — when during training is human feedback most effective (early / mid / late / throughout)?
2. **Magnitude** — how does the scale of the human reward signal affect learning?

---

## Table of Contents

- [Background](#background)
- [Method](#method)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Scripts Reference](#scripts-reference)
  - [run.py — Baseline](#runpy--baseline-training)
  - [train_hcrl.py — Interactive HCRL](#train_hcrlpy--interactive-hcrl)
  - [train_rlhf.py — RLHF (oracle)](#train_rlhfpy--rlhf-oracle)
  - [train_rlhf_human.py — RLHF (human)](#train_rlhf_humanpy--rlhf-human)
  - [feedback_timing_experiment.py](#feedback_timing_experimentpy)
  - [sensitivity_analysis.py](#sensitivity_analysispy)
  - [run_all.py — Full pipeline](#run_allpy--full-automated-pipeline)
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

Human trainer gives real-time scalar feedback (+/−) that is added to the environment reward. Key claims:
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
| Learning rate α | 0.05 (HCRL) · 0.2 (RLHF) |
| Discount factor γ | 0.95 (HCRL) · 1.0 (RLHF) |
| Initial exploration ε₀ | 0.50 |
| Exploration decay | 0.99 per episode |
| Termination penalty | −5,000 |

Q-update rule:
```
Q(s,a) ← Q(s,a) + α · [r_total + γ · max Q(s',a') − Q(s,a)]
```

### HCRL Reward Signal

```
r_total = r_env + r_human          (normal timestep)
r_total = −5000 + r_human          (early termination — preserves human signal)
```

### HCRL Reward Model (`HCRLRewardModel`)

Trained by **MSE regression** on `(observation, human_reward)` pairs collected when the human presses a key. After training it predicts a reward for every state, filling the silent timesteps where the human gave no signal.

```
Loss = (1/N) Σ (r̂(sᵢ) − hᵢ)²     hᵢ ∈ {+10, −10}
```

### RLHF Reward Model (`RewardModel`)

Trained by **cross-entropy on pairwise preferences** (Bradley-Terry model):

```
P̂(A ≻ B) = exp(Σ r̂(sₜᴬ)) / (exp(Σ r̂(sₜᴬ)) + exp(Σ r̂(sₜᴮ)))
Loss     = −[ μ · log P̂(A≻B) + (1−μ) · log P̂(B≻A) ]
```

`μ = 1` if human (or oracle) preferred clip A, `0` if preferred B, `0.5` if tie.

Both reward models share architecture: **2-layer MLP** (obs_dim → 64 → 64 → 1, tanh activations) trained with Adam. Saved as `.npz`.

### Oracle (Automated Human)

For reproducible automated experiments, a simulated oracle replaces the human:

- **HCRL oracle** (`oracle_feedback`) — gives continuous graded feedback proportional to pole angle and cart position stability. Trigger probability: 50% per timestep (models human reaction time).
- **RLHF oracle** (`oracle_preference`) — Boltzmann-rational model that picks the better clip with probability proportional to `exp(score/T)`.

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
├── train_hcrl.py                      # Interactive HCRL (↑/↓ keyboard feedback)
├── train_rlhf.py                      # RLHF with simulated oracle (automated)
├── train_rlhf_human.py                # RLHF with real human clip comparisons (pygame)
│
├── feedback_timing_experiment.py      # Timing experiment: Early/Mid/Late/Full × 3 seeds
├── sensitivity_analysis.py            # Weight sensitivity: [5,20,50] × 3 seeds
├── run_all.py                         # Master pipeline (baseline → timing → sensitivity → analysis)
│
├── compare_models.py                  # Training curves + gameplay + significance tests
├── convergence_analysis.py            # Threshold-crossing convergence speed
├── analyze_feedback.py                # Analyze human feedback logs
│
├── watch.py                           # Watch a single model play (pygame)
├── visual_compare.py                  # Up to 6 models side-by-side (pygame)
├── webapp.py                          # Flask web visualizer (stream frames to browser)
│
├── compare.py                         # Legacy comparison wrapper
├── replay.py                          # Low-level replay utility (used by watch.py)
│
├── tests/
│   ├── test_episode_history.py
│   ├── test_qlearning_agent.py
│   └── test_random_agent.py
│
└── experiment-results/                # All output (models, CSVs, plots)
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

## Quick Start

```bash
# Fully automated: baseline + timing + sensitivity + analysis (no human needed)
uv run python run_all.py --episodes 200

# Watch a trained model
uv run python watch.py experiment-results/ep200/baseline_model.npz

# Interactive HCRL (you press ↑/↓ in the game window)
uv run python train_hcrl.py

# RLHF with oracle (automated, no human)
uv run python train_rlhf.py

# RLHF with real human comparisons
uv run python train_rlhf_human.py
```

---

## Scripts Reference

### `run.py` — Baseline Training

Pure Q-Learning with no human feedback. Trains 3 seeds and saves models and episode histories.

```bash
uv run python run.py
uv run python run.py --episodes 500
uv run python run.py --episodes 200 --verbose
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--episodes` | int | 100 | Training episodes per seed |
| `--verbose` | flag | off | Open render window + live matplotlib plot |

**Output:** `experiment-results/ep{N}/baseline_s{0,1,2}_model.npz`, `baseline_s{0,1,2}_history.csv`

---

### `train_hcrl.py` — Interactive HCRL

Human-in-the-loop training with real keyboard feedback. Opens a CartPole render window.

```bash
uv run python train_hcrl.py
```

**Controls during the feedback window** (first 20% of episodes by default):

| Key | Effect |
|---|---|
| `↑` Arrow Up | +10 reward (good move) |
| `↓` Arrow Down | −10 reward (bad move) |
| `Esc` | Quit training |

**Key constants** (edit in script):

| Constant | Default | Description |
|---|---|---|
| `max_episodes_to_run` | 100 | Total training episodes |
| `max_timesteps_per_episode` | 200 | Max timesteps per episode |
| `terminate_penalty` | 5000 | Penalty for early termination |
| `feedback_window` | first 20% | Episode range where keyboard input is accepted |
| Human reward magnitude | ±10 | Reward given on keypress |

**Output:** `experiment-results/hcrl_model.npz`, `hcrl_episode_history.csv`, `hcrl_feedback_log.csv`, `hcrl_reward_model.npz`

---

### `train_rlhf.py` — RLHF (oracle)

Fully automated RLHF using a simulated oracle (no human needed). Trains a preference-based reward model then uses it to drive Q-learning.

```bash
uv run python train_rlhf.py
```

**Key constants** (edit in script):

| Constant | Default | Description |
|---|---|---|
| `SEED` | 42 | Random seed |
| `SEGMENT_LENGTH` | 25 | Timesteps per trajectory clip |
| `WARMUP_SEGMENTS` | 60 | Initial clips collected before RL |
| `SEGMENTS_PER_ITER` | 10 | New clips added each iteration |
| `PAIRS_PER_ITER` | 32 | Preference queries per reward-model update |
| `REWARD_MODEL_EPOCHS` | 40 | Gradient steps per update |
| `SEGMENT_BUFFER_SIZE` | 500 | Max clips kept in replay buffer |
| `WARMUP_EPISODES` | 50 | Episodes with env reward before RLHF |
| `NUM_ITERATIONS` | 60 | Main RLHF loop iterations |
| `EPISODES_PER_ITER` | 8 | Policy episodes per iteration |
| `REWARD_MODEL_LR` | 3e-4 | Adam learning rate for reward model |
| `REWARD_MODEL_HIDDEN` | 64 | Hidden layer size |
| `AGENT_LR` | 0.2 | Q-learning rate |
| `AGENT_DISCOUNT` | 1.0 | Discount factor γ |
| `AGENT_EXPLORATION` | 0.5 | Initial ε |
| `AGENT_EXPLORATION_DECAY` | 0.99 | ε decay per episode |

**Total episodes:** `WARMUP_EPISODES + NUM_ITERATIONS × EPISODES_PER_ITER` = 530

**Output:** `experiment-results/reward_model.npz`, `experiment-results/rlhf_results.png`

---

### `train_rlhf_human.py` — RLHF (human)

Interactive RLHF: you watch pairs of CartPole clips in a pygame window and press a key to say which looks better. The reward model learns from your labels.

```bash
uv run python train_rlhf_human.py
```

**Controls during labelling:**

| Key | Meaning |
|---|---|
| `A` | Clip A was better (μ = 1.0) |
| `B` | Clip B was better (μ = 0.0) |
| `S` | Skip / tie (μ = 0.5) |
| `Esc` | Stop labelling, save and exit |

**Key constants** (edit in script):

| Constant | Default | Description |
|---|---|---|
| `SEED` | 42 | Random seed |
| `SEGMENT_LENGTH` | 50 | Timesteps per clip shown to human |
| `WARMUP_SEGMENTS` | 20 | Clips collected before first labelling |
| `SEGMENTS_PER_ITER` | 4 | New clips per iteration |
| `SEGMENT_BUFFER_SIZE` | 300 | Max clips in buffer |
| `WARMUP_PAIRS` | 10 | Pairs shown during bootstrap round |
| `PAIRS_PER_ITER` | 4 | Pairs labelled per iteration (keep low to avoid fatigue) |
| `REWARD_MODEL_EPOCHS` | 50 | Gradient steps after each labelling round |
| `WARMUP_EPISODES` | 30 | Episodes with env reward before RLHF |
| `NUM_ITERATIONS` | 20 | Main RLHF loop iterations |
| `EPISODES_PER_ITER` | 8 | Policy episodes per iteration |
| `CLIP_FPS` | 12 | Playback speed when showing clips |
| `REWARD_MODEL_LR` | 3e-4 | Adam learning rate for reward model |
| `REWARD_MODEL_HIDDEN` | 64 | Hidden layer size |

**Total episodes:** `WARMUP_EPISODES + NUM_ITERATIONS × EPISODES_PER_ITER` = 190
**Total clip pairs to label:** `WARMUP_PAIRS + NUM_ITERATIONS × PAIRS_PER_ITER` = 90 (~30–45 min)

**Output:** `experiment-results/reward_model_human.npz`, `experiment-results/rlhf_human_results.png`

---

### `feedback_timing_experiment.py`

Tests when human feedback is most effective. Trains 4 conditions × 3 seeds = 12 runs.

| Condition | Feedback window (N episodes) |
|---|---|
| Early | 0% → 20% |
| Mid | 40% → 60% |
| Late | 80% → 100% |
| Full Feedback | 0% → 100% |

```bash
# Automated (oracle, no keyboard)
uv run python feedback_timing_experiment.py --auto --episodes 200
uv run python feedback_timing_experiment.py --auto --episodes 500

# Interactive (you give feedback via keyboard)
uv run python feedback_timing_experiment.py --episodes 200

# Analyze only (re-generate charts from existing results)
uv run python feedback_timing_experiment.py --analyze --episodes 200
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
- `{early,mid,late,full_feedback}_s{seed}_model.npz`
- `{early,mid,late,full_feedback}_s{seed}_history.csv`
- `{early,mid,late,full_feedback}_s{seed}_feedback_log.csv`
- `timing_experiment_results.png`

Analysis prints **Mann-Whitney U test** (one-sided) for each HCRL condition vs. baseline. Significance: `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns`.

---

### `sensitivity_analysis.py`

Tests how feedback magnitude affects learning. Full Feedback window, 3 seeds × 3 weights = 9 runs.

```bash
uv run python sensitivity_analysis.py --episodes 200
uv run python sensitivity_analysis.py --episodes 500

# Analyze only
uv run python sensitivity_analysis.py --episodes 200 --analyze
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
- `w{weight}_s{seed}_model.npz`, `w{weight}_s{seed}.csv`
- `sensitivity_results.png`

---

### `run_all.py` — Full Automated Pipeline

Runs the complete experiment pipeline in one command: baseline → timing experiment → sensitivity analysis → compare models → convergence analysis. No human input required.

```bash
uv run python run_all.py --episodes 200
uv run python run_all.py --episodes 500

# Skip training, re-run analysis only
uv run python run_all.py --episodes 200 --analyze-only

# Skip chart generation (useful for headless environments)
uv run python run_all.py --episodes 200 --skip-charts
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--episodes` | int | 200 | Training episodes per condition |
| `--eval-episodes` | int | 100 | Gameplay evaluation episodes per model |
| `--analyze-only` | flag | off | Skip all training |
| `--skip-charts` | flag | off | Skip chart steps |

---

### `compare_models.py`

Compare all 5 models (Baseline + 4 HCRL timing conditions). Multi-seed aware: shows mean ± std shaded bands. Runs a Welch's t-test + Cohen's d for each HCRL model vs. baseline.

```bash
uv run python compare_models.py --episodes 200
uv run python compare_models.py --episodes 500
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--episodes` | int | 100 | Which episode-count results to load |
| `--eval-episodes` | int | 100 | Evaluation gameplay episodes per model |

**Output:** `experiment-results/ep{N}/comparison_training.png`, `comparison_gameplay.png`

Effect size interpretation: `|d| < 0.2` negligible · `< 0.5` small · `< 0.8` medium · `≥ 0.8` large.

---

### `convergence_analysis.py`

Measures how fast each model first crosses episode-length thresholds of **50, 100, 150, 200** using a 10-episode rolling mean.

```bash
uv run python convergence_analysis.py --episodes 200
uv run python convergence_analysis.py --episodes 500
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--episodes` | int | 100 | Which episode-count results to load |

**Key constants:**

| Constant | Default | Description |
|---|---|---|
| `THRESHOLDS` | [50, 100, 150, 200] | Performance thresholds to detect |
| `ROLLING_WINDOW` | 10 | Rolling mean window size |

**Output:** `experiment-results/ep{N}/convergence_analysis.png`. Also prints AUC (area under learning curve) — higher = better training efficiency.

---

### `analyze_feedback.py`

Analyze timing, frequency, and state distribution of feedback from interactive HCRL sessions.

```bash
# Single session
uv run python analyze_feedback.py experiment-results/ep200

# Compare all 4 timing conditions
uv run python analyze_feedback.py --compare experiment-results/ep200/timing-experiment
```

**Output:** `feedback_analysis.png`, `conditions_feedback_comparison.png`

---

### `watch.py`

Opens a pygame window and plays a single trained model for N episodes. Prints mean, median, best, and worst lengths.

```bash
uv run python watch.py experiment-results/ep200/baseline_model.npz
uv run python watch.py experiment-results/ep500/timing-experiment/late_model.npz 20
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `model_path` | path | required | Path to `.npz` model file |
| `num_episodes` | int | 10 | Episodes to watch |

---

### `visual_compare.py`

Watch up to **6 models play simultaneously** in a dynamic pygame grid. Grid layout adapts to number of models:

| Models | Layout |
|---|---|
| 1–2 | 1 × 2 |
| 3–4 | 2 × 2 |
| 5–6 | 2 × 3 |

```bash
# Compare 4 timing conditions (2×2 grid)
uv run python visual_compare.py \
  experiment-results/ep500/baseline_model.npz \
  experiment-results/ep500/timing-experiment/early_model.npz \
  experiment-results/ep500/timing-experiment/mid_model.npz \
  experiment-results/ep500/timing-experiment/late_model.npz \
  --labels Baseline "Early (0-20%)" "Mid (40-60%)" "Late (80-100%)" \
  --episodes 10

# Compare sensitivity models (1×3 grid)
uv run python visual_compare.py \
  experiment-results/ep500/sensitivity/w5_s0_model.npz \
  experiment-results/ep500/sensitivity/w20_s0_model.npz \
  experiment-results/ep500/sensitivity/w50_s0_model.npz \
  --labels "Weight=5" "Weight=20" "Weight=50"
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `models` | path(s) | required | 1–6 paths to `.npz` files |
| `--labels` | str(s) | filename stem | Display labels (must match model count) |
| `--episodes` | int | 10 | Episodes to play |

Press `Esc` or close window to stop.

---

### `webapp.py` — Web Visualizer

Watch trained models in a browser — no pygame required.

```bash
uv run python webapp.py
# Open: http://localhost:5000
```

Features:
- Auto-discovers all `.npz` models from `experiment-results/`, grouped by episode count and type
- Episode slider (1–30) and speed slider (5–60 fps)
- Live frame streaming via Server-Sent Events (SSE)
- Per-model stats: current steps, running mean, best episode
- Results table: mean, median, best, worst, ≥195 rate

**Expose publicly with ngrok:**
```bash
# Terminal 1
uv run python webapp.py

# Terminal 2
ngrok http 5000
# Share the https://....ngrok-free.app URL
```

**Optional:** Place `hust_logo.png` in the project root for a header logo.

---

## Output Directory Layout

All outputs go to `experiment-results/`:

```
experiment-results/
│
├── hcrl_model.npz                     # train_hcrl.py
├── hcrl_episode_history.csv
├── hcrl_feedback_log.csv
├── hcrl_reward_model.npz              # HCRLRewardModel weights
│
├── reward_model.npz                   # train_rlhf.py (oracle)
├── rlhf_results.png
│
├── reward_model_human.npz             # train_rlhf_human.py
├── rlhf_human_results.png
│
└── ep{N}/                             # N-episode runs (200, 500, …)
    ├── baseline_s{0,1,2}_model.npz
    ├── baseline_s{0,1,2}_history.csv
    ├── baseline_model.npz             # seed-0 canonical copy
    ├── comparison_training.png
    ├── comparison_gameplay.png
    ├── convergence_analysis.png
    │
    ├── timing-experiment/
    │   ├── {early,mid,late,full_feedback}_s{0,1,2}_model.npz
    │   ├── {early,mid,late,full_feedback}_s{0,1,2}_history.csv
    │   ├── {early,mid,late,full_feedback}_s{0,1,2}_feedback_log.csv
    │   ├── {early,mid,late,full_feedback}_model.npz   # seed-0 canonical copy
    │   └── timing_experiment_results.png
    │
    └── sensitivity/
        ├── w{5,20,50}_s{0,1,2}_model.npz
        ├── w{5,20,50}_s{0,1,2}.csv
        └── sensitivity_results.png
```

---

## Hyperparameter Reference

### HCRL / Baseline Agent

| Parameter | Value | Notes |
|---|---|---|
| Learning rate α | 0.05 | Stable with large termination penalty |
| Discount factor γ | 0.95 | Slightly myopic for sparse env reward |
| Initial exploration ε₀ | 0.50 | 50% random at start |
| Exploration decay | 0.99/ep | ~13% at ep 200, ~0.7% at ep 500 |
| Termination penalty | −5,000 | Strongly discourages early failure |
| State bins | 7/feature | 4,096 total states |
| Oracle trigger probability | 0.50 | Models human reaction time |
| Feedback weight (timing exp.) | 10.0 | Keyboard magnitude |
| Feedback weights (sensitivity) | 5, 20, 50 | — |
| Seeds | 0, 1, 2 | 3 seeds for statistical validity |

### RLHF Reward Model

| Parameter | Value | Notes |
|---|---|---|
| Architecture | MLP 4→64→64→1 | tanh activations |
| Optimiser | Adam | β₁=0.9, β₂=0.999 |
| Learning rate | 3e-4 | Both RLHF and HCRL models |
| RLHF loss | Cross-entropy (Bradley-Terry) | Pairwise preferences |
| HCRL loss | MSE regression | Direct scalar feedback |
| Segment length (oracle) | 25 timesteps | ~2 sec at 12 fps |
| Segment length (human) | 50 timesteps | ~4 sec at 12 fps |

---

## Development

```bash
uv sync                     # install all deps including dev tools

uv run ruff check .         # lint
uv run ty check             # type check
uv run pytest               # tests
```

Pre-commit hooks (black, isort, ruff, ty) run automatically on `git commit` after:
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

**Pygame window freezes during RLHF human labelling:**
Click the window once to bring it into focus before pressing keys.

---

## References

Christiano, P., Leike, J., Brown, T. B., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

Knox, W. B., & Stone, P. (2009). Interactively shaping agents via human reinforcement: The TAMER framework. *Proceedings of the 5th International Conference on Knowledge Capture (K-CAP)*, pp. 9–16. ACM. https://doi.org/10.1145/1597735.1597738

Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983). Neuronlike adaptive elements that can solve difficult learning control problems. *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-13(5), 834–846.

Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *Proceedings of the 16th ICML*, pp. 278–287.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
