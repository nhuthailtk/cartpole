"""
Shared training utilities for all CartPole experiments.

Functions here eliminate the duplicate training-loop implementations that
previously existed across train_hcrl.py, train_vi_tamer.py,
feedback_timing_experiment.py, and sensitivity_analysis.py.

Also provides shared IO helpers (save_history_csv, save_feedback_csv) that
replaced identical copies scattered across training and analysis scripts.
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import asdict
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import pandas as pd

from cartpole import config as cfg
from cartpole.agents import QLearningAgent, VITAMERAgent
from cartpole.entities import EpisodeHistory, EpisodeHistoryRecord

if TYPE_CHECKING:
    from cartpole.reward_model import EnsembleRewardModel, HCRLRewardModel, RewardModel


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def make_agent(rng: np.random.Generator, *, discount: float = cfg.AGENT_DISCOUNT) -> QLearningAgent:
    """Create a QLearningAgent with standard hyperparameters from config."""
    return QLearningAgent(
        learning_rate=cfg.AGENT_LR,
        discount_factor=discount,
        exploration_rate=cfg.AGENT_EXPLORE,
        exploration_decay_rate=cfg.AGENT_DECAY,
        random_state=rng,
    )


def make_vi_tamer_agent(rng: np.random.Generator, *, discount: float = cfg.VI_TAMER_DISCOUNT) -> VITAMERAgent:
    """Create a VITAMERAgent with standard hyperparameters from config."""
    return VITAMERAgent(
        learning_rate=cfg.AGENT_LR,
        discount_factor=discount,
        exploration_rate=cfg.AGENT_EXPLORE,
        exploration_decay_rate=cfg.AGENT_DECAY,
        random_state=rng,
    )


# ---------------------------------------------------------------------------
# HCRL / TAMER episode runner  [HCRL §III-A  — Knox & Stone 2009]
# ---------------------------------------------------------------------------

def run_hcrl_episode(
    env: gym.Env,
    agent: QLearningAgent,
    reward_model: HCRLRewardModel | None,
    rng: np.random.Generator,
    *,
    feedback_weight: float = cfg.HCRL_FEEDBACK_WEIGHT,
    in_feedback_window: bool = True,
    terminate_penalty: float = cfg.HCRL_TERMINATE_PENALTY,
    oracle_fn=None,
) -> tuple[int, list[np.ndarray], list[float], list[dict]]:
    """
    Run one HCRL episode.

    The oracle fires with HCRL_TRIGGER_PROB at each timestep when
    `in_feedback_window` is True.  The reward model fills silent timesteps.
    Env reward is the fallback when the model has not been trained yet.

    Parameters
    ----------
    oracle_fn : callable(obs, weight, rng) -> float, defaults to oracle_feedback

    Returns
    -------
    episode_length  : int
    new_obs         : list of observations where oracle fired
    new_rewards     : list of oracle signals (paired with new_obs)
    feedback_log    : list of dicts (one per oracle signal)
    """
    from cartpole.oracle import oracle_feedback
    oracle_fn = oracle_fn or oracle_feedback

    model_ready = reward_model is not None

    ep_obs:    list[np.ndarray] = []
    ep_rew:    list[float]      = []
    fb_log:    list[dict]       = []
    t0 = time.time()

    obs, _ = env.reset()
    action = agent.begin_episode(obs)

    for t in range(cfg.MAX_TIMESTEPS):
        next_obs, env_reward, terminated, truncated, _ = env.step(action)

        oracle_signal: float = 0.0
        if in_feedback_window:
            oracle_signal = oracle_fn(obs, feedback_weight, rng)

        if oracle_signal != 0.0:
            ep_obs.append(obs.copy())
            ep_rew.append(oracle_signal)
            shaped = oracle_signal
            fb_log.append({
                "timestamp":     time.time() - t0,
                "episode_step":  t,
                "feedback":      "positive" if oracle_signal > 0 else "negative",
                "magnitude":     abs(oracle_signal),
                "reward":        oracle_signal,
                "cart_position": float(obs[0]),
                "cart_velocity": float(obs[1]),
                "pole_angle":    float(obs[2]),
                "pole_velocity": float(obs[3]),
            })
        elif model_ready:
            shaped = float(reward_model.predict(obs))
        else:
            shaped = float(env_reward)

        is_successful = t >= cfg.MAX_TIMESTEPS - 1
        if terminated and not is_successful:
            shaped -= terminate_penalty

        action = agent.act(next_obs, shaped)
        obs = next_obs

        if terminated or truncated or is_successful:
            return t + 1, ep_obs, ep_rew, fb_log

    return cfg.MAX_TIMESTEPS, ep_obs, ep_rew, fb_log


def run_vi_tamer_episode(
    env: gym.Env,
    agent: VITAMERAgent,
    reward_model: HCRLRewardModel | None,
    rng: np.random.Generator,
    *,
    feedback_weight: float = cfg.HCRL_FEEDBACK_WEIGHT,
    in_feedback_window: bool = True,
    terminate_penalty: float = cfg.HCRL_TERMINATE_PENALTY,
) -> tuple[int, list[np.ndarray], list[float], list[dict]]:
    """
    Run one VI-TAMER episode using act_vi() for non-myopic TD updates.

    Same signature as run_hcrl_episode() so callers can be swapped.
    """
    from cartpole.oracle import oracle_feedback

    model_ready = reward_model is not None

    ep_obs:    list[np.ndarray] = []
    ep_rew:    list[float]      = []
    fb_log:    list[dict]       = []
    t0 = time.time()

    obs, _ = env.reset()
    action = agent.begin_episode(obs)

    for t in range(cfg.MAX_TIMESTEPS):
        next_obs, env_reward, terminated, truncated, _ = env.step(action)

        oracle_signal: float = 0.0
        if in_feedback_window:
            oracle_signal = oracle_feedback(obs, feedback_weight, rng)

        if oracle_signal != 0.0:
            ep_obs.append(obs.copy())
            ep_rew.append(oracle_signal)
            fb_log.append({
                "timestamp":     time.time() - t0,
                "episode_step":  t,
                "feedback":      "positive" if oracle_signal > 0 else "negative",
                "magnitude":     abs(oracle_signal),
                "reward":        oracle_signal,
                "cart_position": float(obs[0]),
                "cart_velocity": float(obs[1]),
                "pole_angle":    float(obs[2]),
                "pole_velocity": float(obs[3]),
            })

        is_successful = t >= cfg.MAX_TIMESTEPS - 1
        penalty = terminate_penalty if (terminated and not is_successful) else 0.0

        # VI-TAMER non-myopic TD update
        action = agent.act_vi(
            obs=obs,
            next_obs=next_obs,
            reward_model=reward_model if model_ready else None,
            env_reward=float(env_reward) - penalty,
        )
        obs = next_obs

        if terminated or truncated or is_successful:
            return t + 1, ep_obs, ep_rew, fb_log

    return cfg.MAX_TIMESTEPS, ep_obs, ep_rew, fb_log


# ---------------------------------------------------------------------------
# RLHF episode / segment helpers  [RLHF §2.2 — Christiano et al. 2017]
# ---------------------------------------------------------------------------

def run_rl_episode(
    env: gym.Env,
    agent: QLearningAgent,
    reward_model: RewardModel | EnsembleRewardModel | None = None,
    *,
    normalise: bool = False,
) -> int:
    """Run one episode. Returns episode length."""
    obs, _ = env.reset()
    action = agent.begin_episode(obs)
    steps = 0
    while True:
        next_obs, env_reward, terminated, truncated, _ = env.step(action)
        if reward_model is not None:
            reward = (
                reward_model.predict_normalised(next_obs)
                if normalise
                else reward_model.predict(next_obs)
            )
        else:
            reward = float(env_reward)
        action = agent.act(next_obs, reward)
        obs = next_obs
        steps += 1
        if terminated or truncated:
            return steps


def collect_segment(
    env: gym.Env,
    agent: QLearningAgent,
    rng: np.random.Generator,
    reward_model: RewardModel | EnsembleRewardModel | None = None,
    *,
    normalise: bool = False,
    seg_length: int = cfg.RLHF_SEGMENT_LENGTH,
) -> np.ndarray:
    """Collect a fixed-length trajectory segment. Returns (seg_length, obs_dim) array."""
    obs_buf: list[np.ndarray] = []
    obs, _ = env.reset()
    action = agent.begin_episode(obs)

    while len(obs_buf) < seg_length:
        obs_buf.append(obs.copy())
        next_obs, env_reward, terminated, truncated, _ = env.step(action)
        if reward_model is not None:
            reward = (
                reward_model.predict_normalised(next_obs)
                if normalise
                else reward_model.predict(next_obs)
            )
        else:
            reward = float(env_reward)
        action = agent.act(next_obs, reward)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            action = agent.begin_episode(obs)

    return np.array(obs_buf)


def sample_preference_pairs(
    segment_buffer: list[np.ndarray],
    n_pairs: int,
    rng: np.random.Generator,
    *,
    error_prob: float = 0.0,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """
    Randomly sample segment pairs and query oracle for preferences.

    Parameters
    ----------
    error_prob : human error rate passed to oracle_preference (0 = no noise)
    """
    from cartpole.reward_model import oracle_preference

    segs_a, segs_b, prefs = [], [], []
    indices = list(range(len(segment_buffer)))
    for _ in range(n_pairs):
        i, j = rng.choice(indices, size=2, replace=False)
        mu = oracle_preference(segment_buffer[i], segment_buffer[j], rng,
                               error_prob=error_prob)
        segs_a.append(segment_buffer[i])
        segs_b.append(segment_buffer[j])
        prefs.append(mu)
    return segs_a, segs_b, prefs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(
    agent: QLearningAgent,
    num_episodes: int = 100,
) -> list[int]:
    """
    Run a trained agent with no step limit and return episode lengths.

    Used by all comparison and analysis scripts to assess final policy quality.
    The agent's exploration rate is NOT reset — call with a fully trained agent
    (exploration_rate ≈ 0 after many episodes of decay).
    """
    env = gym.make("CartPole-v1", max_episode_steps=None)
    episode_lengths: list[int] = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        action = agent.begin_episode(obs)
        steps = 0
        while True:
            obs, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                episode_lengths.append(steps)
                break
            action = agent.act(obs, reward=0.0)
    env.close()
    return episode_lengths


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def rolling_mean(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean via convolution."""
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def save_history_csv(
    episode_lengths: list[int],
    path: str | pathlib.Path,
) -> pathlib.Path:
    """Save a list of episode lengths to a CSV with column 'episode_length'."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"episode_length": episode_lengths}).to_csv(p, index_label="episode_index")
    print(f"History saved to {p}")
    return p


def save_episode_history_csv(
    history: EpisodeHistory,
    path: str | pathlib.Path,
) -> pathlib.Path:
    """Save an EpisodeHistory object to CSV (preserves all dataclass fields)."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    records = (asdict(r) for r in history.all_records())
    pd.DataFrame.from_records(records, index="episode_index").to_csv(p, header=True)
    print(f"Episode history saved to {p}")
    return p


def save_feedback_csv(
    feedback_log: list[dict],
    path: str | pathlib.Path,
) -> pathlib.Path:
    """Save a list of feedback event dicts to CSV."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(feedback_log).to_csv(p, index=False)
    print(f"Feedback log saved to {p} ({len(feedback_log)} events)")
    return p


def make_episode_history() -> EpisodeHistory:
    """Create an EpisodeHistory with the standard success criterion from config."""
    return EpisodeHistory(
        max_timesteps_per_episode=cfg.MAX_TIMESTEPS,
        goal_avg_episode_length=cfg.GOAL_LENGTH,
        goal_consecutive_episodes=cfg.GOAL_CONSECUTIVE,
    )
