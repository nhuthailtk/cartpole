"""
Baseline Q-Learning training for CartPole.

Pure tabular Q-Learning with no human feedback — serves as the control
condition against which HCRL and RLHF results are compared.

Usage
-----
    uv run python run.py
    uv run python run.py --episodes 200
"""

import argparse
import pathlib
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import numpy as np
import pandas as pd

from cartpole import config as cfg
from cartpole.agents import QLearningAgent
from cartpole.entities import Action, EpisodeHistory, EpisodeHistoryRecord, Observation, Reward
from cartpole.train_utils import make_agent, save_episode_history_csv


def run_agent(
    agent: QLearningAgent,
    env: gym.Env,
    verbose: bool = False,
    max_episodes: int = 100,
) -> EpisodeHistory:
    """Run the baseline agent for max_episodes and return the episode history."""
    episode_history = EpisodeHistory(
        max_timesteps_per_episode=cfg.MAX_TIMESTEPS,
        goal_avg_episode_length=cfg.GOAL_LENGTH,
        goal_consecutive_episodes=cfg.GOAL_CONSECUTIVE,
    )
    episode_history_plotter = None

    if verbose:
        from cartpole.plotting import EpisodeHistoryMatplotlibPlotter
        episode_history_plotter = EpisodeHistoryMatplotlibPlotter(
            history=episode_history, visible_episode_count=200,
        )
        episode_history_plotter.create_plot()

    print("Running the environment. To stop, press Ctrl-C.")
    try:
        for episode_index in range(max_episodes):
            observation, _ = env.reset()
            action = agent.begin_episode(observation)

            for timestep_index in range(cfg.MAX_TIMESTEPS):
                observation, step_reward, terminated, _, _ = env.step(action)
                reward: Reward = float(step_reward)

                if verbose:
                    _log_timestep(timestep_index, action, reward, observation)

                is_successful = timestep_index >= cfg.MAX_TIMESTEPS - 1
                if terminated and not is_successful:
                    reward = float(-cfg.BASELINE_TERMINATE_PENALTY)

                action = agent.act(observation, reward)

                if terminated or is_successful:
                    print(f"Episode {episode_index} finished after {timestep_index + 1} timesteps.")
                    episode_history.record_episode(
                        EpisodeHistoryRecord(
                            episode_index=episode_index,
                            episode_length=timestep_index + 1,
                            is_successful=is_successful,
                        )
                    )
                    if verbose and episode_history_plotter:
                        episode_history_plotter.update_plot()
                    if episode_history.is_goal_reached():
                        print(f"SUCCESS: Goal reached after {episode_index + 1} episodes!")
                        return episode_history
                    break

        print(f"FAILURE: Goal not reached after {max_episodes} episodes.")

    except KeyboardInterrupt:
        print("WARNING: Terminated by user request.")

    return episode_history


def _log_timestep(index: int, action: Action, reward: Reward, observation: Observation) -> None:
    print(
        f"Timestep: {index:3d}   Action: {action:2d}   Reward: {reward:5.1f}   "
        f"Cart Pos: {observation[0]:6.3f}   Cart Vel: {observation[1]:6.3f}   "
        f"Angle: {observation[2]:6.3f}   Tip Vel: {observation[3]:6.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline Q-Learning training")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_dir = pathlib.Path(f"experiment-results/ep{args.episodes}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_lengths = []
    for seed in cfg.SEEDS:
        print(f"\n--- Baseline seed={seed} ---")
        rng = np.random.default_rng(seed=seed)
        env = gym.make("CartPole-v1", render_mode="human" if args.verbose else None)
        agent = make_agent(rng)
        history = run_agent(agent=agent, env=env, verbose=args.verbose, max_episodes=args.episodes)
        env.close()

        hist_path = out_dir / f"baseline_s{seed}_history.csv"
        save_episode_history_csv(history, hist_path)
        agent.save(out_dir / f"baseline_s{seed}_model.npz")

        lengths = [r.episode_length for r in history.all_records()]
        all_lengths.append(lengths)
        print(f"  seed={seed}: mean={pd.Series(lengths).mean():.1f}, "
              f"last-30={pd.Series(lengths).tail(30).mean():.1f}")

    # Canonical seed-0 copies for backward compatibility with downstream scripts
    shutil.copy(out_dir / "baseline_s0_history.csv", out_dir / "episode_history.csv")
    shutil.copy(out_dir / "baseline_s0_model.npz",   out_dir / "baseline_model.npz")

    means   = [pd.Series(l).mean()         for l in all_lengths]
    last30s = [pd.Series(l).tail(30).mean() for l in all_lengths]
    print(f"\nBaseline ({len(cfg.SEEDS)} seeds):")
    print(f"  Overall mean : {pd.Series(means).mean():.1f} ± {pd.Series(means).std():.1f}")
    print(f"  Last-30 avg  : {pd.Series(last30s).mean():.1f} ± {pd.Series(last30s).std():.1f}")


if __name__ == "__main__":
    main()
