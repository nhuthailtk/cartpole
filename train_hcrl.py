"""
HCRL training — Knox & Stone (2009).

"Interactively Shaping Agents via Human Reinforcement: The TAMER Framework"
Knox, W. B., & Stone, P. (K-CAP 2009)

Two feedback modes:
  oracle (default) — simulated oracle fires at HCRL_TRIGGER_PROB per step.
  human  (--human) — real human presses arrow keys while watching the agent.

Usage
-----
    uv run python train_hcrl.py                          # oracle (automated)
    uv run python train_hcrl.py --human                  # real human feedback
    uv run python train_hcrl.py --episodes 200 --seed 0
    uv run python train_hcrl.py --human --episodes 100 --seed 0

Controls (--human mode)
-----------------------
  [Arrow Up]   — positive feedback  (+10)
  [Arrow Down] — negative feedback  (−10)
  [Esc]        — quit early
"""

import argparse
import pathlib
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame

from cartpole import config as cfg
from cartpole.agents import QLearningAgent
from cartpole.entities import EpisodeHistory, EpisodeHistoryRecord
from cartpole.reward_model import HCRLRewardModel
from cartpole.train_utils import (
    make_agent,
    run_hcrl_episode,
    save_episode_history_csv,
    save_feedback_csv,
    save_history_csv,
)


# ---------------------------------------------------------------------------
# Interactive helper — kept for feedback_timing_experiment.py (human mode)
# ---------------------------------------------------------------------------

def run_hcrl_agent(
    agent: QLearningAgent,
    env: gym.Env,
    verbose: bool = False,
    feedback_window: tuple[int, int] | None = None,
    reward_model: HCRLRewardModel | None = None,
) -> tuple[EpisodeHistory, list[dict]]:
    """
    Interactive HCRL: human provides real-time feedback via arrow keys.

    Used by feedback_timing_experiment.py when running in interactive mode.
    Automated oracle training uses train() below.
    """
    max_episodes  = 100
    fb_start, fb_end = feedback_window if feedback_window else (0, int(0.2 * max_episodes))

    episode_history = EpisodeHistory(
        max_timesteps_per_episode=cfg.MAX_TIMESTEPS,
        goal_avg_episode_length=cfg.GOAL_LENGTH,
        goal_consecutive_episodes=cfg.GOAL_CONSECUTIVE,
    )
    feedback_log:    list[dict]       = []
    rm_obs_buf:      list[np.ndarray] = []
    rm_reward_buf:   list[float]      = []

    if verbose:
        from cartpole.plotting import EpisodeHistoryMatplotlibPlotter
        plotter = EpisodeHistoryMatplotlibPlotter(history=episode_history, visible_episode_count=200)
        plotter.create_plot()
    else:
        plotter = None

    print("=" * 58)
    print(" HCRL TRAINING — Arrow Up: +reward  |  Arrow Down: -reward")
    print(f" Feedback window: Episode {fb_start} → {fb_end}")
    print("=" * 58)

    pygame.init()
    t0 = time.time()
    try:
        for ep in range(max_episodes):
            obs, _ = env.reset()
            action  = agent.begin_episode(obs)
            ep_fb   = 0

            for t in range(cfg.MAX_TIMESTEPS):
                human_reward = 0.0
                in_window = fb_start <= ep < fb_end

                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP and in_window:
                            human_reward = cfg.HCRL_FEEDBACK_WEIGHT
                            ep_fb += 1
                            if reward_model is not None:
                                rm_obs_buf.append(obs.copy())
                                rm_reward_buf.append(human_reward)
                            feedback_log.append({
                                "timestamp": time.time() - t0, "episode": ep,
                                "timestep": t, "feedback": "positive",
                                "reward": human_reward,
                                "cart_position": float(obs[0]), "cart_velocity": float(obs[1]),
                                "pole_angle": float(obs[2]),    "pole_velocity": float(obs[3]),
                            })
                        elif event.key == pygame.K_DOWN and in_window:
                            human_reward = -cfg.HCRL_FEEDBACK_WEIGHT
                            ep_fb += 1
                            if reward_model is not None:
                                rm_obs_buf.append(obs.copy())
                                rm_reward_buf.append(human_reward)
                            feedback_log.append({
                                "timestamp": time.time() - t0, "episode": ep,
                                "timestep": t, "feedback": "negative",
                                "reward": human_reward,
                                "cart_position": float(obs[0]), "cart_velocity": float(obs[1]),
                                "pole_angle": float(obs[2]),    "pole_velocity": float(obs[3]),
                            })
                        elif event.key == pygame.K_ESCAPE:
                            raise KeyboardInterrupt

                if in_window:
                    time.sleep(0.1)

                next_obs, step_reward, terminated, _, _ = env.step(action)

                shaped: float
                if human_reward != 0.0:
                    shaped = human_reward
                elif reward_model is not None and len(rm_obs_buf) >= 2:
                    shaped = float(reward_model.predict(obs))
                else:
                    shaped = float(step_reward)

                is_successful = t >= cfg.MAX_TIMESTEPS - 1
                if terminated and not is_successful:
                    shaped -= cfg.HCRL_TERMINATE_PENALTY

                action = agent.act(next_obs, shaped)
                obs = next_obs

                if terminated or is_successful:
                    print(f"Episode {ep} done — {t + 1} steps, {ep_fb} feedbacks")
                    time.sleep(0.5)
                    episode_history.record_episode(EpisodeHistoryRecord(
                        episode_index=ep, episode_length=t + 1, is_successful=is_successful,
                    ))
                    if plotter:
                        plotter.update_plot()
                    if reward_model is not None and len(rm_obs_buf) >= 2:
                        reward_model.train_on_feedback(
                            np.array(rm_obs_buf), np.array(rm_reward_buf)
                        )
                    if episode_history.is_goal_reached():
                        print(f"SUCCESS after {ep + 1} episodes!")
                        return episode_history, feedback_log
                    break

        print(f"FAILURE: Goal not reached after {max_episodes} episodes.")
    except KeyboardInterrupt:
        print("Terminated by user.")
    finally:
        pygame.quit()

    return episode_history, feedback_log


# Backward-compatible aliases used by feedback_timing_experiment.py
def save_feedback_log(feedback_log: list[dict], experiment_dir: str,
                      filename: str = "hcrl_feedback_log.csv") -> pathlib.Path:
    return save_feedback_csv(feedback_log, pathlib.Path(experiment_dir) / filename)


def save_history(history: EpisodeHistory, experiment_dir: str,
                 filename: str = "hcrl_episode_history.csv") -> pathlib.Path:
    return save_episode_history_csv(history, pathlib.Path(experiment_dir) / filename)


# ---------------------------------------------------------------------------
# Oracle training (automated)
# ---------------------------------------------------------------------------

def train(total_episodes: int, seed: int, feedback_weight: float = cfg.HCRL_FEEDBACK_WEIGHT) -> None:
    """Train HCRL with simulated oracle feedback (no human required)."""
    out = pathlib.Path(cfg.experiment_dir(total_episodes, f"hcrl-oracle-fw{feedback_weight:g}"))
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  HCRL (oracle)  —  {total_episodes} episodes  seed={seed}")
    print(f"  Oracle trigger prob : {cfg.HCRL_TRIGGER_PROB}")
    print(f"  Feedback weight     : ±{feedback_weight}")
    print(f"  Output: {out}")
    print("=" * 60)

    rng   = np.random.default_rng(seed)
    env   = gym.make("CartPole-v1")
    agent = make_agent(rng)
    reward_model = HCRLRewardModel(obs_dim=env.observation_space.shape[0])

    episode_lengths: list[int]        = []
    rm_losses:       list[float]      = []
    rm_obs_buf:      list[np.ndarray] = []
    rm_reward_buf:   list[float]      = []
    total_feedback   = 0
    model_ready      = False

    print(f"\n{'Episode':>8} {'Length':>7} {'Avg10':>7} {'FB_total':>9} {'RM loss':>9}")
    print("-" * 46)

    for ep in range(total_episodes):
        ep_len, new_obs, new_rew, _ = run_hcrl_episode(
            env, agent, reward_model if model_ready else None, rng,
            in_feedback_window=True,
            feedback_weight=feedback_weight,
        )
        episode_lengths.append(ep_len)
        rm_obs_buf.extend(new_obs)
        rm_reward_buf.extend(new_rew)
        total_feedback += len(new_obs)

        if len(rm_obs_buf) >= 2:
            loss = reward_model.train_on_feedback(
                np.array(rm_obs_buf), np.array(rm_reward_buf),
            )
            rm_losses.append(loss)
            model_ready = True
            loss_str = f"{loss:9.4f}"
        else:
            loss_str = "      n/a"

        if (ep + 1) % max(1, total_episodes // 10) == 0 or ep == 0:
            avg10 = np.mean(episode_lengths[-10:])
            print(f"  {ep+1:6d}  {ep_len:6d}  {avg10:7.1f}  {total_feedback:8d}  {loss_str}")

    env.close()

    agent.save(out / f"hcrl_oracle_s{seed}_model.npz")
    reward_model.save(out / f"hcrl_oracle_s{seed}_reward_model.npz")
    save_history_csv(episode_lengths, out / f"hcrl_oracle_s{seed}_history.csv")
    print(f"\nSaved to {out}/  |  Total oracle feedback: {total_feedback}")

    _plot(episode_lengths, rm_losses, total_feedback,
          f"HCRL (oracle) — {total_episodes} eps, seed={seed}, {total_feedback} signals",
          "forestgreen", out / f"hcrl_oracle_s{seed}_results.png")


# ---------------------------------------------------------------------------
# Human training (interactive)
# ---------------------------------------------------------------------------

def train_human(total_episodes: int, seed: int, feedback_weight: float = cfg.HCRL_FEEDBACK_WEIGHT) -> None:
    """Train HCRL with real human feedback via arrow keys."""
    out = pathlib.Path(cfg.experiment_dir(total_episodes, f"hcrl-human-fw{feedback_weight:g}"))
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  HCRL (human feedback)  —  {total_episodes} episodes  seed={seed}")
    print(f"  Output: {out}")
    print("=" * 60)
    print()
    print("  Controls  (click the CartPole window first):")
    print("  [Arrow Up]   = positive feedback  +10  (good!)")
    print("  [Arrow Down] = negative feedback  -10  (bad!)")
    print("  [Esc]        = quit early")
    print()

    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1", render_mode="human")
    agent = make_agent(rng)
    reward_model = HCRLRewardModel(obs_dim=env.observation_space.shape[0])

    pygame.init()

    episode_lengths: list[int]        = []
    rm_losses:       list[float]      = []
    rm_obs_buf:      list[np.ndarray] = []
    rm_reward_buf:   list[float]      = []
    total_feedback   = 0
    model_ready      = False

    try:
        for ep in range(total_episodes):
            obs, _ = env.reset()
            action = agent.begin_episode(obs)
            ep_feedback = 0
            quit_requested = False

            for t in range(cfg.MAX_TIMESTEPS):
                human_reward = 0.0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_requested = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            quit_requested = True
                        elif event.key == pygame.K_UP:
                            human_reward = feedback_weight
                            ep_feedback += 1
                            total_feedback += 1
                            rm_obs_buf.append(obs.copy())
                            rm_reward_buf.append(human_reward)
                            print(f"  [+] ep={ep+1:3d} t={t:3d}  angle={obs[2]:+.3f}  cart={obs[0]:+.2f}")
                        elif event.key == pygame.K_DOWN:
                            human_reward = -feedback_weight
                            ep_feedback += 1
                            total_feedback += 1
                            rm_obs_buf.append(obs.copy())
                            rm_reward_buf.append(human_reward)
                            print(f"  [-] ep={ep+1:3d} t={t:3d}  angle={obs[2]:+.3f}  cart={obs[0]:+.2f}")

                if quit_requested:
                    break

                next_obs, env_reward, terminated, truncated, _ = env.step(action)

                if human_reward != 0.0:
                    shaped = human_reward
                elif model_ready:
                    shaped = float(reward_model.predict(obs))
                else:
                    shaped = float(env_reward)

                if terminated and t < cfg.MAX_TIMESTEPS - 1:
                    shaped -= cfg.HCRL_TERMINATE_PENALTY

                action = agent.act(next_obs, shaped)
                obs = next_obs

                time.sleep(0.05)

                if terminated or truncated:
                    break

            ep_len = t + 1
            episode_lengths.append(ep_len)

            if quit_requested:
                print(f"\n  Quit at episode {ep + 1}.")
                break

            if len(rm_obs_buf) >= 2:
                loss = reward_model.train_on_feedback(
                    np.array(rm_obs_buf), np.array(rm_reward_buf),
                )
                rm_losses.append(loss)
                model_ready = True
                loss_str = f"rm_loss={loss:.4f}"
            else:
                loss_str = "rm_loss=n/a"

            avg10 = np.mean(episode_lengths[-10:])
            print(f"  ep {ep+1:4d}/{total_episodes}  len={ep_len:3d}  avg10={avg10:6.1f}"
                  f"  fb_ep={ep_feedback}  fb_total={total_feedback}  {loss_str}")

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        env.close()
        pygame.quit()

    if not episode_lengths:
        print("No episodes completed — nothing to save.")
        return

    agent.save(out / f"hcrl_human_s{seed}_model.npz")
    reward_model.save(out / f"hcrl_human_s{seed}_reward_model.npz")
    save_history_csv(episode_lengths, out / f"hcrl_human_s{seed}_history.csv")
    print(f"\nSaved to {out}/  |  Total human feedback: {total_feedback}")

    _plot(episode_lengths, rm_losses, total_feedback,
          f"HCRL (human) — {len(episode_lengths)} eps, seed={seed}, {total_feedback} signals",
          "mediumseagreen", out / f"hcrl_human_s{seed}_results.png")


# ---------------------------------------------------------------------------
# Shared plot helper
# ---------------------------------------------------------------------------

def _plot(
    episode_lengths: list[int],
    rm_losses: list[float],
    total_feedback: int,
    title: str,
    color: str,
    save_path: pathlib.Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13)

    ax = axes[0]
    lengths = np.array(episode_lengths)
    ax.plot(lengths, alpha=0.3, color=color)
    if len(lengths) >= 20:
        rm = np.convolve(lengths, np.ones(20) / 20, mode="valid")
        ax.plot(range(19, len(lengths)), rm, color=color, linewidth=2, label="Rolling mean (20)")
    ax.axhline(cfg.GOAL_LENGTH, color="gray", linestyle="--", alpha=0.5,
               label=f"Goal: {cfg.GOAL_LENGTH}")
    ax.set(xlabel="Episode", ylabel="Length", title="Policy performance")
    ax.legend(fontsize=8)
    ax.set_ylim(0, cfg.MAX_TIMESTEPS + 10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if rm_losses:
        ax.plot(rm_losses, color="darkorange", marker="o", markersize=3)
    ax.set(xlabel="Episode", ylabel="Reward model MSE loss",
           title=f"R_H learning ({total_feedback} signals)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Plot saved to {save_path}")
    plt.show()


def save_history_csv_simple(episode_lengths: list[int], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"episode_length": episode_lengths}).to_csv(path, index_label="episode_index")
    print(f"History saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HCRL training (oracle or human feedback)")
    parser.add_argument("--episodes",        type=int,   default=100)
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--feedback-weight", type=float, default=cfg.HCRL_FEEDBACK_WEIGHT,
                        help="Magnitude of +/- oracle reward signal (default: %(default)s)")
    parser.add_argument("--human",    action="store_true",
                        help="Use real human arrow-key feedback instead of simulated oracle")
    args = parser.parse_args()

    if args.human:
        train_human(args.episodes, args.seed, args.feedback_weight)
    else:
        train(args.episodes, args.seed, args.feedback_weight)
