"""
VI-TAMER training — Knox & Stone (2012).

"Reinforcement Learning from Human Reward and Advice" — Knox & Stone (2012)
Survey: Li et al. (2019), Section III-A-2 "Learning From Nonmyopic Human Reward"

VI-TAMER extends TAMER by adding a value function Q_H(s,a) driven by R_H,
enabling non-myopic planning through human reward signals.

  TAMER    (γ=0):  policy = argmax_a R_H(s,a)         [immediate only]
  VI-TAMER (γ>0):  Q_H(s,a) <- Q_H + α·(R_H + γ·max Q_H(s') - Q_H)
                   policy    = argmax_a Q_H(s,a)        [discounted future]

Two feedback modes:
  oracle (default) — simulated oracle fires at HCRL_TRIGGER_PROB per step.
  human  (--human) — real human presses arrow keys while watching the agent.

Usage
-----
    uv run python train_vi_tamer.py                          # oracle (automated)
    uv run python train_vi_tamer.py --human                  # real human feedback
    uv run python train_vi_tamer.py --gamma 0                # recovers plain TAMER
    uv run python train_vi_tamer.py --episodes 200 --seed 0
    uv run python train_vi_tamer.py --human --episodes 100 --seed 0

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
import pygame

from cartpole import config as cfg
from cartpole.reward_model import HCRLRewardModel
from cartpole.train_utils import (
    make_vi_tamer_agent,
    run_vi_tamer_episode,
    save_history_csv,
)


# ---------------------------------------------------------------------------
# Oracle training (automated)
# ---------------------------------------------------------------------------

def train(total_episodes: int, seed: int, gamma: float, feedback_weight: float = cfg.HCRL_FEEDBACK_WEIGHT) -> None:
    out = pathlib.Path(cfg.experiment_dir(total_episodes, f"vi-tamer-fw{feedback_weight:g}"))
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  VI-TAMER  —  {total_episodes} episodes  seed={seed}  γ={gamma}")
    print(f"  Oracle trigger prob : {cfg.HCRL_TRIGGER_PROB}")
    print(f"  Feedback weight     : ±{feedback_weight}")
    print(f"  Output: {out}")
    print("=" * 60)
    print(f"\n  TAMER (γ=0): policy = argmax_a R_H(s,a) [immediate]")
    print(f"  VI-TAMER  (γ={gamma}): Q_H bootstraps future value via TD\n")

    rng   = np.random.default_rng(seed)
    env   = gym.make("CartPole-v1")
    agent = make_vi_tamer_agent(rng, discount=gamma)
    reward_model = HCRLRewardModel(obs_dim=env.observation_space.shape[0])

    episode_lengths: list[int]        = []
    rm_losses:       list[float]      = []
    rm_obs_buf:      list[np.ndarray] = []
    rm_reward_buf:   list[float]      = []
    total_feedback   = 0
    model_ready      = False

    print(f"{'Episode':>8} {'Length':>7} {'Avg10':>7} {'FB_total':>9} {'RM loss':>9}")
    print("-" * 46)

    for ep in range(total_episodes):
        ep_len, new_obs, new_rew, _ = run_vi_tamer_episode(
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

    agent.save(out / f"vi_tamer_s{seed}_model.npz")
    reward_model.save(out / f"vi_tamer_s{seed}_reward_model.npz")
    save_history_csv(episode_lengths, out / f"vi_tamer_s{seed}_history.csv")
    print(f"\nSaved to {out}/  |  Total oracle feedback: {total_feedback}")

    _plot(episode_lengths, rm_losses, total_feedback, gamma,
          f"VI-TAMER (γ={gamma}) — {total_episodes} eps, seed={seed}",
          "steelblue", out / f"vi_tamer_s{seed}_results.png")


# ---------------------------------------------------------------------------
# Human training (interactive)
# ---------------------------------------------------------------------------

def train_human(total_episodes: int, seed: int, gamma: float, feedback_weight: float = cfg.HCRL_FEEDBACK_WEIGHT) -> None:
    """Train VI-TAMER with real human feedback via arrow keys."""
    out = pathlib.Path(cfg.experiment_dir(total_episodes, f"vi-tamer-human-fw{feedback_weight:g}"))
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  VI-TAMER (human)  —  {total_episodes} episodes  seed={seed}  γ={gamma}")
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
    agent = make_vi_tamer_agent(rng, discount=gamma)
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

                # Compute the shaped signal (human > model > env fallback)
                if human_reward != 0.0:
                    shaped = human_reward
                elif model_ready:
                    shaped = float(reward_model.predict(obs))
                else:
                    shaped = float(env_reward)

                if terminated and t < cfg.MAX_TIMESTEPS - 1:
                    shaped -= cfg.HCRL_TERMINATE_PENALTY

                # VI-TAMER non-myopic update: pass pre-computed signal as env_reward,
                # no reward_model so act_vi uses it directly.
                action = agent.act_vi(
                    obs=obs,
                    next_obs=next_obs,
                    reward_model=None,
                    env_reward=shaped,
                )
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

    agent.save(out / f"vi_tamer_human_s{seed}_model.npz")
    reward_model.save(out / f"vi_tamer_human_s{seed}_reward_model.npz")
    save_history_csv(episode_lengths, out / f"vi_tamer_human_s{seed}_history.csv")
    print(f"\nSaved to {out}/  |  Total human feedback: {total_feedback}")

    _plot(episode_lengths, rm_losses, total_feedback, gamma,
          f"VI-TAMER human (γ={gamma}) — {len(episode_lengths)} eps, seed={seed}",
          "cornflowerblue", out / f"vi_tamer_human_s{seed}_results.png")


# ---------------------------------------------------------------------------
# Shared plot helper
# ---------------------------------------------------------------------------

def _plot(
    episode_lengths: list[int],
    rm_losses: list[float],
    total_feedback: int,
    gamma: float,
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
    ax.set(xlabel="Episode", ylabel="Length", title=f"VI-TAMER policy (γ={gamma})")
    ax.legend(fontsize=8)
    ax.set_ylim(0, cfg.MAX_TIMESTEPS + 10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if rm_losses:
        ax.plot(rm_losses, color="darkorange", marker="o", markersize=3)
    ax.set(xlabel="Episode", ylabel="MSE loss",
           title=f"R_H learning ({total_feedback} signals)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VI-TAMER training (oracle or human feedback)")
    parser.add_argument("--episodes",        type=int,   default=100)
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--gamma",           type=float, default=cfg.VI_TAMER_DISCOUNT,
                        help="Discount factor γ (0 = plain TAMER)")
    parser.add_argument("--feedback-weight", type=float, default=cfg.HCRL_FEEDBACK_WEIGHT,
                        help="Magnitude of +/- oracle reward signal (default: %(default)s)")
    parser.add_argument("--human",    action="store_true",
                        help="Use real human arrow-key feedback instead of simulated oracle")
    args = parser.parse_args()

    if args.human:
        train_human(args.episodes, args.seed, args.gamma, args.feedback_weight)
    else:
        train(args.episodes, args.seed, args.gamma, args.feedback_weight)
