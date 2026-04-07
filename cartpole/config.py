"""
Centralized configuration for all CartPole experiments.

All hyperparameters from both papers are defined here so every script
imports constants from one place instead of redefining them.

Papers:
  [HCRL]  Li et al., "Human-Centered Reinforcement Learning: A Survey", 2019
           Knox & Stone, "Interactively Shaping Agents via Human Reinforcement
           (TAMER)", K-CAP 2009
           Knox & Stone, "Reinforcement Learning from Human Reward and Advice"
           (VI-TAMER), AAMAS 2012

  [RLHF]  Christiano et al., "Deep Reinforcement Learning from Human
           Preferences", NeurIPS 2017
"""

# ---------------------------------------------------------------------------
# CartPole-v1 environment constants (Gymnasium specification)
# ---------------------------------------------------------------------------

CARTPOLE_ANGLE_LIMIT:    float = 0.2095   # radians — episode termination boundary
CARTPOLE_POSITION_LIMIT: float = 2.4      # metres  — episode termination boundary
MAX_TIMESTEPS:           int   = 200      # maximum steps per episode

# Success criterion: mean episode length ≥ GOAL_LENGTH over GOAL_CONSECUTIVE episodes
GOAL_LENGTH:      int = 195
GOAL_CONSECUTIVE: int = 30

# ---------------------------------------------------------------------------
# Agent hyperparameters — identical across ALL methods for fair comparison
# ---------------------------------------------------------------------------

AGENT_LR:       float = 0.05   # Q-table learning rate (α)
AGENT_DISCOUNT: float = 0.95   # discount factor (γ) — also used by VI-TAMER for Q_H
AGENT_EXPLORE:  float = 0.50   # initial ε-greedy exploration rate
AGENT_DECAY:    float = 0.99   # per-episode exploration decay multiplier
NUM_BINS:       int   = 7      # discretisation bins per continuous state feature

# ---------------------------------------------------------------------------
# HCRL / TAMER parameters  [HCRL §III-A "Interactive Shaping"]
# ---------------------------------------------------------------------------
# Knox & Stone (2009): oracle fires at each timestep with ORACLE_TRIGGER_PROB
# to model human reaction time limitations.

HCRL_TRIGGER_PROB:  float = 0.50   # probability oracle fires per timestep
HCRL_FEEDBACK_WEIGHT: float = 10.0  # magnitude of +/- reward signal
HCRL_TERMINATE_PENALTY: float = 50.0  # penalty added when episode ends early
HCRL_RM_LR:     float = 1e-3       # HCRLRewardModel Adam learning rate
HCRL_RM_HIDDEN: int   = 64         # hidden layer width
HCRL_RM_EPOCHS: int   = 20         # gradient steps per reward-model update

# ---------------------------------------------------------------------------
# Baseline terminate penalty (pure Q-Learning, env-reward scale ~1/step)
# ---------------------------------------------------------------------------

BASELINE_TERMINATE_PENALTY: float = 5000.0

# ---------------------------------------------------------------------------
# VI-TAMER parameters  [HCRL §III-A-2 "Learning From Nonmyopic Human Reward"]
# ---------------------------------------------------------------------------
# Knox & Stone (2012): AGENT_DISCOUNT (γ > 0) propagates future value through
# Q_H(s,a). γ = 0 recovers plain TAMER.
# VI-TAMER uses the same reward model as TAMER (HCRLRewardModel).

VI_TAMER_DISCOUNT: float = AGENT_DISCOUNT  # default γ; override with --gamma

# ---------------------------------------------------------------------------
# RLHF parameters  [RLHF §2.2 "Our Method"]
# ---------------------------------------------------------------------------

RLHF_SEGMENT_LENGTH:  int   = 40     # timesteps per trajectory clip shown to human
RLHF_WARMUP_FRACTION: float = 0.20   # fraction of total episodes used as warm-up
RLHF_EPISODES_PER_ITER: int = 8      # policy episodes per RLHF iteration
RLHF_WARMUP_SEGMENTS: int   = 40     # segments collected during warm-up
RLHF_SEGMENTS_PER_ITER: int = 8      # new segments collected per iteration
RLHF_PAIRS_PER_ITER:  int   = 8      # preference queries per reward-model update
RLHF_RM_EPOCHS:       int   = 40     # gradient steps per reward-model update
RLHF_SEGMENT_BUFFER:  int   = 400    # rolling segment replay buffer size
RLHF_RM_LR:           float = 3e-4   # RewardModel Adam learning rate
RLHF_RM_HIDDEN:       int   = 64     # hidden layer width

# ---------------------------------------------------------------------------
# Ensemble RLHF parameters  [RLHF §2.2 modifications]
# ---------------------------------------------------------------------------
# §2.2 (bullet 1): ensemble of K predictors trained on bootstrapped subsets
# §2.2.4: uncertainty-based query selection via ensemble variance
# §2.2.1: reward normalisation to zero mean / unit std
# §2.2.3: human error modelled as 10% uniform-random response

ENSEMBLE_N_MODELS:        int   = 3               # ensemble size K
ENSEMBLE_CANDIDATES_MULT: int   = 10              # candidate pairs = mult × PAIRS_PER_ITER
ENSEMBLE_ERROR_PROB:      float = 0.1             # oracle human-error rate (§2.2.3)
ENSEMBLE_VAL_FRACTION:    float = 1.0 / 2.718281828  # validation split ≈ 1/e

# ---------------------------------------------------------------------------
# Oracle parameters (shared across HCRL and RLHF)
# ---------------------------------------------------------------------------
# Stability scoring weights used by oracle_feedback() and oracle_preference()

ORACLE_ANGLE_WEIGHT:    float = 0.7   # weight of pole-angle stability in score
ORACLE_POSITION_WEIGHT: float = 0.3   # weight of cart-position stability in score
ORACLE_TEMPERATURE:     float = 0.05  # Boltzmann rationality for RLHF oracle

# HCRL oracle thresholds (discrete good / bad / silent)
ORACLE_GOOD_ANGLE_STAB: float = 0.55   # angle_stab above this → clearly good
ORACLE_GOOD_POS_STAB:   float = 0.20   # pos_stab above this → clearly good
ORACLE_BAD_ANGLE_STAB:  float = 0.35   # angle_stab below this → clearly bad
ORACLE_BAD_POS_STAB:    float = 0.25   # pos_stab below this → clearly bad

# ---------------------------------------------------------------------------
# Experiment design
# ---------------------------------------------------------------------------

SEEDS: list[int] = [0, 1, 2]

# Feedback magnitude sensitivity study  [HCRL §III-B inspired]
SENSITIVITY_WEIGHTS: list[float] = [5.0, 20.0, 50.0]

# Feedback timing conditions (as fractions of total episodes)  [HCRL §IV]
TIMING_CONDITIONS: dict[str, tuple[float, float]] = {
    "early":         (0.00, 0.20),
    "mid":           (0.40, 0.60),
    "late":          (0.80, 1.00),
    "full_feedback": (0.00, 1.00),
}

# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def experiment_dir(total_episodes: int, method: str) -> str:
    """Return standard output directory path for a given method and episode count."""
    return f"experiment-results/ep{total_episodes}/{method}"
