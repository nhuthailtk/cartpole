"""
HCRL training with simulated oracle feedback — Knox & Stone (2009).

"Interactively Shaping Agents via Human Reinforcement: The TAMER Framework"
Knox, W. B., & Stone, P. (K-CAP 2009)

Automated version of train_hcrl_human.py.  A simulated oracle calls
oracle_feedback() at each timestep (50% trigger probability) to produce
+/- signals based on the CartPole state.  A HCRLRewardModel (MLP regression)
learns from those signals and predicts rewards when the oracle is silent.

Pipeline
--------
1. Each episode: run agent in a headless CartPole env.
2. oracle_feedback() fires with probability ORACLE_TRIGGER_PROB each step.
   - +FEEDBACK_WEIGHT when pole is stable and cart centred
   - -FEEDBACK_WEIGHT when pole is near failure or cart is near edge
   - 0.0 (silent) otherwise
3. Reward signal each step:
   - oracle signal  if oracle fired
   - reward model prediction  if oracle silent and model trained
   - env reward (+1/step)  if oracle silent and model not yet trained
4. After each episode: retrain HCRLRewardModel on all collected (obs, reward) pairs.
5. Save agent + reward model + history + plot.

Usage
-----
    uv run python train_hcrl.py
    uv run python train_hcrl.py --episodes 200 --seed 0
    uv run python train_hcrl.py --episodes 500 --seed 1

Note
----
The helper functions run_hcrl_agent / save_feedback_log / save_history are
kept for backward compatibility with feedback_timing_experiment.py.
"""

import argparse
import pathlib
import sys
import time
from dataclasses import asdict

sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame

from cartpole.agents import Agent, QLearningAgent
from cartpole.entities import Action, EpisodeHistory, EpisodeHistoryRecord, Observation, Reward
from cartpole.oracle import oracle_feedback
from cartpole.reward_model import HCRLRewardModel

# ---------------------------------------------------------------------------
# Agent hyper-parameters — identical to baseline / RLHF for fair comparison
# ---------------------------------------------------------------------------
AGENT_LR                = 0.05
AGENT_DISCOUNT          = 0.95
AGENT_EXPLORATION       = 0.5
AGENT_EXPLORATION_DECAY = 0.99

# Oracle / feedback
ORACLE_TRIGGER_PROB     = 0.5    # matches cartpole/oracle.py default
FEEDBACK_WEIGHT         = 10.0
TERMINATE_PENALTY       = 50.0

# Reward model
REWARD_MODEL_LR         = 1e-3
REWARD_MODEL_HIDDEN     = 64
REWARD_MODEL_EPOCHS     = 20

MAX_TIMESTEPS           = 200


# ---------------------------------------------------------------------------
# Backward-compatible helpers (used by feedback_timing_experiment.py)
# ---------------------------------------------------------------------------

def run_hcrl_agent(
    agent: Agent,
    env: gym.Env,
    verbose: bool = False,
    feedback_window: tuple[int, int] | None = None,
    reward_model: HCRLRewardModel | None = None,
) -> tuple[EpisodeHistory, list[dict]]:
    """
    Run the agent with Human-in-the-loop reward shaping (TAMER approach).
    The user can press Up Arrow for positive feedback and Down Arrow for negative feedback.
    Returns (episode_history, feedback_log).

    feedback_window: (start_ep, end_ep) — only accept human feedback during these episodes.
                     If None, feedback is accepted for the first 20 episodes (default behavior).

    Implementation based on the TAMER framework:
        Knox, W. B., & Stone, P. (2009). Interactively shaping agents via human
        reinforcement: The TAMER framework. Proceedings of the 5th International
        Conference on Knowledge Capture (K-CAP), pp. 9-16. ACM.
        https://doi.org/10.1145/1597735.1597738
    """
    max_episodes_to_run = 100
    max_timesteps_per_episode = 200

    terminate_penalty = 5000

    goal_avg_episode_length = 195
    goal_consecutive_episodes = 30

    episode_history = EpisodeHistory(
        max_timesteps_per_episode=200,
        goal_avg_episode_length=goal_avg_episode_length,
        goal_consecutive_episodes=goal_consecutive_episodes,
    )
    episode_history_plotter = None

    if verbose:
        from cartpole.plotting import EpisodeHistoryMatplotlibPlotter

        episode_history_plotter = EpisodeHistoryMatplotlibPlotter(
            history=episode_history,
            visible_episode_count=200,
        )
        episode_history_plotter.create_plot()

    # Determine feedback window (default: first 20% of episodes)
    if feedback_window is None:
        fb_start, fb_end = 0, int(0.2 * max_episodes_to_run)
    else:
        fb_start, fb_end = feedback_window

    print("==========================================================")
    print(" HUMAN-CENTERED REINFORCEMENT LEARNING (HCRL) TRAINING")
    print(" Instructions:")
    print(" - Watch the agent in the game window.")
    print(" - Press [Arrow Up]   ⬆️: Reward (Good move) (+10)")
    print(" - Press [Arrow Down] ⬇️: Penalize (Bad move) (-10)")
    print(" - Press [Esc]: Quit")
    print(f" Feedback window: Episode {fb_start} → {fb_end}")
    print("==========================================================")

    # Feedback log: record every human keystroke
    feedback_log: list[dict] = []
    training_start_time = time.time()

    # Reward model: accumulate (obs, reward) pairs for training
    rm_obs_buf:    list[np.ndarray] = []
    rm_reward_buf: list[float]      = []

    # Initialize pygame to capture keyboard events
    pygame.init()

    try:
        for episode_index in range(max_episodes_to_run):
            observation, _ = env.reset()
            action = agent.begin_episode(observation)

            # Reset feedback signal tracker for printing
            episode_feedback_count = 0

            for timestep_index in range(max_timesteps_per_episode):

                # --- [HCRL COMPONENT] Listen for Human Feedback ---
                human_reward = 0.0
                in_feedback_window = fb_start <= episode_index < fb_end

                # We pump pygame events to catch keystrokes when the debug window is active
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP and in_feedback_window:
                            human_reward = 10.0
                            print(f"[HCRL] Positive feedback! Timestep {timestep_index}")
                            episode_feedback_count += 1
                            if reward_model is not None:
                                rm_obs_buf.append(observation.copy())
                                rm_reward_buf.append(human_reward)
                            feedback_log.append({
                                "timestamp": time.time() - training_start_time,
                                "episode": episode_index,
                                "timestep": timestep_index,
                                "feedback": "positive",
                                "reward": human_reward,
                                "cart_position": float(observation[0]),
                                "cart_velocity": float(observation[1]),
                                "pole_angle": float(observation[2]),
                                "pole_velocity": float(observation[3]),
                            })
                        elif event.key == pygame.K_DOWN and in_feedback_window:
                            human_reward = -10.0
                            print(f"[HCRL] Negative feedback! Timestep {timestep_index}")
                            episode_feedback_count += 1
                            if reward_model is not None:
                                rm_obs_buf.append(observation.copy())
                                rm_reward_buf.append(human_reward)
                            feedback_log.append({
                                "timestamp": time.time() - training_start_time,
                                "episode": episode_index,
                                "timestep": timestep_index,
                                "feedback": "negative",
                                "reward": human_reward,
                                "cart_position": float(observation[0]),
                                "cart_velocity": float(observation[1]),
                                "pole_angle": float(observation[2]),
                                "pole_velocity": float(observation[3]),
                            })
                        elif event.key == pygame.K_ESCAPE:
                            raise KeyboardInterrupt

                # Optional: slow down the frame rate so the human has time to react and feedback
                if in_feedback_window:
                    time.sleep(0.1)  # Slow down so human has time to react

                # Step the environment
                observation, step_reward, terminated, _, _ = env.step(action)

                # Reward model fills in signal when human is silent
                if reward_model is not None and human_reward == 0.0:
                    shaped_reward = reward_model.predict(observation)
                else:
                    shaped_reward = human_reward

                # Penalize early termination, preserve shaped signal
                is_successful = timestep_index >= max_timesteps_per_episode - 1
                if terminated and not is_successful:
                    total_reward: Reward = float(-terminate_penalty) + shaped_reward
                else:
                    total_reward = float(step_reward) + shaped_reward
                if human_reward != 0:
                    print(f"Total reward: {total_reward}")



                # Agent learns from combined reward (environment + human)
                action = agent.act(observation, total_reward)

                if in_feedback_window:
                    time.sleep(0.1)  # Slow down so human has time to react

                if terminated or is_successful:
                    print(
                        f"Episode {episode_index} "
                        f"finished after {timestep_index + 1} timesteps. "
                        f"(Human feedbacks provided: {episode_feedback_count})"
                    )
                    time.sleep(0.5)

                    episode_history.record_episode(
                        EpisodeHistoryRecord(
                            episode_index=episode_index,
                            episode_length=timestep_index + 1,
                            is_successful=is_successful,
                        )
                    )

                    if verbose and episode_history_plotter:
                        episode_history_plotter.update_plot()

                    # Retrain reward model on all collected feedback so far
                    if reward_model is not None and len(rm_obs_buf) >= 2:
                        obs_arr = np.array(rm_obs_buf)
                        rew_arr = np.array(rm_reward_buf)
                        loss = reward_model.train_on_feedback(obs_arr, rew_arr)
                        print(f"  [RewardModel] trained on {len(rm_obs_buf)} samples, loss={loss:.4f}")

                    if episode_history.is_goal_reached():
                        print(f"SUCCESS: Goal reached after {episode_index + 1} episodes!")
                        return episode_history, feedback_log

                    break

        print(f"FAILURE: Goal not reached after {max_episodes_to_run} episodes.")

    except KeyboardInterrupt:
        print("WARNING: Terminated by user request.")
    finally:
        pygame.quit()  # Release pygame resources

    return episode_history, feedback_log


def save_feedback_log(feedback_log: list[dict], experiment_dir: str, filename: str = "hcrl_feedback_log.csv") -> pathlib.Path:
    """Save the human feedback log to a CSV file."""
    dir_path = pathlib.Path(experiment_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / filename
    df = pd.DataFrame(feedback_log)
    df.to_csv(file_path, index=False)
    print(f"Feedback log saved to {file_path} ({len(feedback_log)} feedbacks)")
    return file_path


def save_history(history: EpisodeHistory, experiment_dir: str, filename: str = "hcrl_episode_history.csv") -> pathlib.Path:
    experiment_dir_path = pathlib.Path(experiment_dir)
    experiment_dir_path.mkdir(parents=True, exist_ok=True)
    file_path = experiment_dir_path / filename
    record_dicts = (asdict(record) for record in history.all_records())
    dataframe = pd.DataFrame.from_records(record_dicts, index="episode_index")
    dataframe.to_csv(file_path, header=True)
    print(f"HCRL Episode history saved to {file_path}")
    return file_path


# ---------------------------------------------------------------------------
# Automated oracle training
# ---------------------------------------------------------------------------

def train(total_episodes: int, seed: int) -> None:
    """Train with a simulated oracle (no human required)."""
    output_dir = pathlib.Path("experiment-results") / f"ep{total_episodes}" / "hcrl-oracle"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  HCRL (oracle)  —  {total_episodes} episodes  seed={seed}")
    print(f"  Oracle trigger prob: {ORACLE_TRIGGER_PROB}")
    print(f"  Feedback weight:     ±{FEEDBACK_WEIGHT}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")

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

    episode_lengths: list[int]   = []
    rm_losses:       list[float] = []
    total_feedback   = 0
    model_ready      = False

    # Growing buffer: all (obs, oracle_reward) pairs collected so far
    rm_obs_buf:    list[np.ndarray] = []
    rm_reward_buf: list[float]      = []

    print(f"\n{'Episode':>8} {'Length':>7} {'Avg10':>7} {'FB_total':>9} {'RM loss':>9}")
    print("-" * 46)

    for ep in range(total_episodes):
        obs, _ = env.reset()
        action = agent.begin_episode(obs)

        for t in range(MAX_TIMESTEPS):
            next_obs, env_reward, terminated, truncated, _ = env.step(action)

            # Oracle fires with ORACLE_TRIGGER_PROB probability
            oracle_signal = oracle_feedback(obs, FEEDBACK_WEIGHT, rng,
                                            trigger_prob=ORACLE_TRIGGER_PROB)

            if oracle_signal != 0.0:
                total_feedback += 1
                rm_obs_buf.append(obs.copy())
                rm_reward_buf.append(oracle_signal)
                shaped = oracle_signal
            elif model_ready:
                shaped = float(reward_model.predict(obs))
            else:
                shaped = env_reward   # fallback before model has data

            if terminated and t < MAX_TIMESTEPS - 1:
                shaped -= TERMINATE_PENALTY

            action = agent.act(next_obs, shaped)
            obs = next_obs

            if terminated or truncated:
                break

        ep_len = t + 1
        episode_lengths.append(ep_len)

        # Retrain reward model after every episode
        if len(rm_obs_buf) >= 2:
            obs_arr = np.array(rm_obs_buf)
            rew_arr = np.array(rm_reward_buf)
            loss = reward_model.train_on_feedback(obs_arr, rew_arr, epochs=REWARD_MODEL_EPOCHS)
            rm_losses.append(loss)
            model_ready = True
            loss_str = f"{loss:9.4f}"
        else:
            loss_str = "      n/a"

        if (ep + 1) % max(1, total_episodes // 10) == 0 or ep == 0:
            avg10 = np.mean(episode_lengths[-10:])
            print(f"  {ep+1:6d}  {ep_len:6d}  {avg10:7.1f}  {total_feedback:8d}  {loss_str}")

    env.close()

    # ── Save ──────────────────────────────────────────────────────────────
    agent.save(output_dir / f"hcrl_oracle_s{seed}_model.npz")
    reward_model.save(output_dir / f"hcrl_oracle_s{seed}_reward_model.npz")
    pd.DataFrame({"episode_length": episode_lengths}).to_csv(
        output_dir / f"hcrl_oracle_s{seed}_history.csv", index_label="episode_index"
    )
    print(f"\nSaved to {output_dir}/")
    print(f"Total oracle feedback signals: {total_feedback}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"HCRL (oracle) — {total_episodes} episodes, seed={seed}, {total_feedback} signals",
        fontsize=13,
    )

    ax = axes[0]
    lengths = np.array(episode_lengths)
    ax.plot(lengths, alpha=0.3, color="forestgreen")
    window = 20
    if len(lengths) >= window:
        rolling = np.convolve(lengths, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window - 1, len(lengths)), rolling,
            color="forestgreen", linewidth=2, label=f"Rolling mean ({window})"
        )
    ax.axhline(195, color="gray", linestyle="--", alpha=0.5, label="Goal: 195")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Length (timesteps)")
    ax.set_title("Policy performance")
    ax.legend(fontsize=8)
    ax.set_ylim(0, MAX_TIMESTEPS + 10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if rm_losses:
        ax.plot(rm_losses, color="darkorange", marker="o", markersize=3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward model MSE loss")
    ax.set_title(f"Reward model ({total_feedback} oracle signals)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"hcrl_oracle_s{seed}_results.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Plot saved to {plot_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HCRL training with simulated oracle feedback"
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
