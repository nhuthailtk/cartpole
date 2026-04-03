import pathlib
import sys
import time
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import pandas as pd
import pygame

from cartpole.agents import Agent, QLearningAgent
from cartpole.entities import Action, EpisodeHistory, EpisodeHistoryRecord, Observation, Reward
from cartpole.reward_model import HCRLRewardModel


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


def main() -> None:
    # Always turn on verbose for HCRL so human can see and interact
    verbose = True
    random_state = np.random.default_rng(seed=0)

    # render_mode must be human
    env = gym.make("CartPole-v1", render_mode="human")
    
    agent = QLearningAgent(
        learning_rate=0.05,
        discount_factor=0.95,
        exploration_rate=0.5,
        exploration_decay_rate=0.99,
        random_state=random_state,
    )

    reward_model = HCRLRewardModel(obs_dim=4, hidden_dim=64, lr=1e-3)

    episode_history, feedback_log = run_hcrl_agent(
        agent=agent, env=env, verbose=verbose, reward_model=reward_model
    )
    save_history(episode_history, experiment_dir="experiment-results")
    save_feedback_log(feedback_log, experiment_dir="experiment-results")
    agent.save("experiment-results/hcrl_model.npz")
    reward_model.save("experiment-results/hcrl_reward_model.npz")


if __name__ == "__main__":
    main()
