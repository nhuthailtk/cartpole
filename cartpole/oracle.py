"""
Oracle (simulated human) feedback for HCRL experiments.

Shared by feedback_timing_experiment.py and sensitivity_analysis.py.
"""

import numpy as np

# Probability of giving feedback at any given timestep.
# Models human reaction time limitations (Knox & Stone, 2009).
ORACLE_TRIGGER_PROB: float = 0.5


def oracle_feedback(
    observation: np.ndarray,
    weight: float,
    rng: np.random.Generator,
    trigger_prob: float = ORACLE_TRIGGER_PROB,
) -> float:
    """
    Simulated human oracle based on visible cart-pole state.
    Models human evaluator behaviour as described in TAMER (Knox & Stone, 2009).
    Trigger probability simulates human reaction time limitations.

    Mimics discrete human judgement — humans don't give graded scores,
    they react when something is clearly good or clearly bad:

      - GOOD   (both stable):    angle_stability > 0.5 AND position_stability > 0.5
                                 → |pole_angle| < ~0.10 rad (~6°), |cart_x| < ~1.2
      - BAD    (either unstable): angle_stability < 0.2 OR position_stability < 0.15
                                 → |pole_angle| > ~0.17 rad (~10°) OR |cart_x| > ~2.0
      - SILENT otherwise         → 0  (ambiguous state, human withholds judgement)

    Parameters
    ----------
    observation : np.ndarray
        CartPole state [cart_x, cart_velocity, pole_angle, pole_velocity].
    weight : float
        Magnitude of positive/negative feedback.
    rng : np.random.Generator
        Random number generator (for trigger probability).
    trigger_prob : float
        Probability of reacting at this timestep (default: ORACLE_TRIGGER_PROB).

    Returns
    -------
    float
        +weight (good), -weight (bad), or 0.0 (silent).
    """
    if rng.random() > trigger_prob:
        return 0.0

    pole_angle = abs(float(observation[2]))
    cart_x     = abs(float(observation[0]))

    # Normalised stability: 1.0 = perfectly centred, 0.0 = at failure boundary
    angle_stability    = max(0.0, 1.0 - pole_angle / 0.2095)   # limit: ±0.2095 rad
    position_stability = max(0.0, 1.0 - cart_x / 2.4)          # limit: ±2.4

    if angle_stability > 0.55 and position_stability > 0.20:
        return weight       # pole < ~5°, cart < ~1.4  → clearly good
    elif angle_stability < 0.35 or position_stability < 0.25:
        return -weight      # pole > ~8.5° OR cart > ~1.9 → clearly bad
    return 0.0
