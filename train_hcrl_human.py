"""
HCRL training with real human feedback — Knox & Stone (2009).

Interactive version: the agent runs in a live CartPole window.
You press arrow keys to give real-time scalar feedback while watching.
A reward model (MLP regression) is trained on your feedback and fills
in predictions when you are silent.

Controls during training
------------------------
  [Arrow Up]   — positive feedback  (+10, "good move!")
  [Arrow Down] — negative feedback  (−10, "bad move!")
  [Esc]        — quit training early

Pipeline
--------
1. Each episode: agent runs in the visible CartPole window.
2. Human presses ↑ / ↓ at any timestep to give +/- feedback.
3. When silent: reward model predicts the reward from the current state.
   (Before the model has enough data, environment reward is used as fallback.)
4. Agent learns Q-values from the combined reward signal each step.
5. After each episode: reward model is retrained on all accumulated feedback.
6. Save agent + reward model + history + plot.

Usage
-----
    uv run python train_hcrl_human.py
    uv run python train_hcrl_human.py --episodes 100 --seed 0
    uv run python train_hcrl_human.py --episodes 200 --seed 1
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

from cartpole.agents import QLearningAgent
from cartpole.reward_model import HCRLRewardModel

# ---------------------------------------------------------------------------
# Agent hyper-parameters — identical to baseline / RLHF for fair comparison
# ---------------------------------------------------------------------------
AGENT_LR                = 0.05
AGENT_DISCOUNT          = 0.95
AGENT_EXPLORATION       = 0.5
AGENT_EXPLORATION_DECAY = 0.99

# Reward model
REWARD_MODEL_LR         = 1e-3
REWARD_MODEL_HIDDEN     = 64
REWARD_MODEL_EPOCHS     = 20

# Feedback
FEEDBACK_WEIGHT         = 10.0   # magnitude of +/- key-press signal
TERMINATE_PENALTY       = 50.0   # extra penalty signal on early episode end
STEP_DELAY              = 0.05   # seconds per step — gives time to react

MAX_TIMESTEPS           = 200


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(total_episodes: int, seed: int) -> None:
    output_dir = pathlib.Path("experiment-results") / f"ep{total_episodes}" / "hcrl-human"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  HCRL (human feedback)  —  {total_episodes} episodes  seed={seed}")
    print(f"  Output: {output_dir}")
    print("=" * 60)
    print()
    print("  Controls  (click the CartPole window first):")
    print("  [Arrow Up]   = positive feedback  +10  (good!)")
    print("  [Arrow Down] = negative feedback  −10  (bad!)")
    print("  [Esc]        = quit early")
    print()

    rng = np.random.default_rng(seed)

    # CartPole with human render so you can watch the agent
    env = gym.make("CartPole-v1", render_mode="human")

    agent = QLearningAgent(
        learning_rate=AGENT_LR,
        discount_factor=AGENT_DISCOUNT,
        exploration_rate=AGENT_EXPLORATION,
        exploration_decay_rate=AGENT_EXPLORATION_DECAY,
        random_state=rng,
    )

    reward_model = HCRLRewardModel(
        obs_dim=env.observation_space.shape[0],
        hidden_dim=REWARD_MODEL_HIDDEN,
        lr=REWARD_MODEL_LR,
    )

    # Gymnasium's human render mode initialises pygame internally.
    # We call init() here to ensure the event queue is active.
    pygame.init()

    episode_lengths: list[int]   = []
    rm_losses:       list[float] = []
    total_feedback:  int         = 0

    # Growing buffer: all (obs, human_reward) pairs seen so far
    rm_obs_buf:    list[np.ndarray] = []
    rm_reward_buf: list[float]      = []

    model_ready = False   # True once reward model has been trained at least once

    try:
        for ep in range(total_episodes):
            obs, _ = env.reset()
            action = agent.begin_episode(obs)
            ep_feedback = 0
            quit_requested = False

            for t in range(MAX_TIMESTEPS):
                # ── Human keyboard events ─────────────────────────────────
                human_reward = 0.0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_requested = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            quit_requested = True
                        elif event.key == pygame.K_UP:
                            human_reward = FEEDBACK_WEIGHT
                            ep_feedback += 1
                            total_feedback += 1
                            rm_obs_buf.append(obs.copy())
                            rm_reward_buf.append(human_reward)
                            print(
                                f"  [+] ep={ep+1:3d} t={t:3d}"
                                f"  angle={obs[2]:+.3f}  cart={obs[0]:+.2f}"
                            )
                        elif event.key == pygame.K_DOWN:
                            human_reward = -FEEDBACK_WEIGHT
                            ep_feedback += 1
                            total_feedback += 1
                            rm_obs_buf.append(obs.copy())
                            rm_reward_buf.append(human_reward)
                            print(
                                f"  [-] ep={ep+1:3d} t={t:3d}"
                                f"  angle={obs[2]:+.3f}  cart={obs[0]:+.2f}"
                            )

                if quit_requested:
                    break

                # ── Step environment ──────────────────────────────────────
                next_obs, env_reward, terminated, truncated, _ = env.step(action)

                # ── Reward signal ─────────────────────────────────────────
                if human_reward != 0.0:
                    # Human spoke — use their signal directly
                    shaped = human_reward
                elif model_ready:
                    # Human silent — let the model predict
                    shaped = float(reward_model.predict(obs))
                else:
                    # Model not trained yet — fall back to env reward (+1/step)
                    shaped = env_reward

                # Strong penalty on early failure
                if terminated and t < MAX_TIMESTEPS - 1:
                    shaped -= TERMINATE_PENALTY

                action = agent.act(next_obs, shaped)
                obs = next_obs

                time.sleep(STEP_DELAY)

                if terminated or truncated:
                    break

            ep_len = t + 1
            episode_lengths.append(ep_len)

            if quit_requested:
                print(f"\n  Quit requested at episode {ep + 1}.")
                break

            # ── Retrain reward model after each episode ───────────────────
            if len(rm_obs_buf) >= 2:
                obs_arr = np.array(rm_obs_buf)
                rew_arr = np.array(rm_reward_buf)
                loss = reward_model.train_on_feedback(
                    obs_arr, rew_arr, epochs=REWARD_MODEL_EPOCHS
                )
                rm_losses.append(loss)
                model_ready = True
                loss_str = f"rm_loss={loss:.4f}"
            else:
                loss_str = "rm_loss=n/a (collecting feedback…)"

            avg10 = np.mean(episode_lengths[-10:])
            print(
                f"  ep {ep+1:4d}/{total_episodes}"
                f"  len={ep_len:3d}"
                f"  avg10={avg10:6.1f}"
                f"  fb_ep={ep_feedback}  fb_total={total_feedback}"
                f"  {loss_str}"
            )

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        env.close()
        pygame.quit()

    if not episode_lengths:
        print("No episodes completed — nothing to save.")
        return

    # ── Save ──────────────────────────────────────────────────────────────
    agent.save(output_dir / f"hcrl_human_s{seed}_model.npz")
    reward_model.save(output_dir / f"hcrl_human_s{seed}_reward_model.npz")
    pd.DataFrame({"episode_length": episode_lengths}).to_csv(
        output_dir / f"hcrl_human_s{seed}_history.csv", index_label="episode_index"
    )
    print(f"\nSaved to {output_dir}/")
    print(f"Total human feedback signals: {total_feedback}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"HCRL (human) — {len(episode_lengths)} episodes, seed={seed}"
        f", {total_feedback} feedback signals",
        fontsize=13,
    )

    ax = axes[0]
    lengths = np.array(episode_lengths)
    ax.plot(lengths, alpha=0.3, color="mediumseagreen")
    window = 20
    if len(lengths) >= window:
        rolling = np.convolve(lengths, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window - 1, len(lengths)), rolling,
            color="mediumseagreen", linewidth=2, label=f"Rolling mean ({window})"
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Length (timesteps)")
    ax.set_title("Policy performance")
    ax.legend(fontsize=8)
    ax.set_ylim(0, MAX_TIMESTEPS + 10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if rm_losses:
        ax.plot(rm_losses, color="darkorange", marker="o", markersize=3)
    ax.set_xlabel("Episode (after first feedback)")
    ax.set_ylabel("Reward model MSE loss")
    ax.set_title(f"Reward model ({total_feedback} feedback signals)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"hcrl_human_s{seed}_results.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Plot saved to {plot_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HCRL training with real human feedback (TAMER-style)"
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Total training episodes (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (default: 0)",
    )
    args = parser.parse_args()
    train(args.episodes, args.seed)
