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


def run_hcrl_agent(agent: Agent, env: gym.Env, verbose: bool = False) -> EpisodeHistory:
    """
    Run the agent with Human-in-the-loop reward shaping (TAMER approach).
    The user can press Up Arrow for positive feedback and Down Arrow for negative feedback.
    """
    max_episodes_to_run = 5000
    max_timesteps_per_episode = 200

    goal_avg_episode_length = 195
    goal_consecutive_episodes = 100

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

    print("==========================================================")
    print(" BẮT ĐẦU CHƯƠNG TRÌNH HỌC TĂNG CƯỜNG VỚI CON NGƯỜI (HCRL)")
    print(" Hướng dẫn:")
    print(" - Xem Agent hoạt động trên cửa sổ game.")
    print(" - Nhấn [Mũi tên Lên] ⬆️: Thưởng (Good move) (+10)")
    print(" - Nhấn [Mũi tên Xuống] ⬇️: Phạt (Bad move)  (-10)")
    print(" - Nhấn [Esc]: Thoát")
    print("==========================================================")

    # Khởi tạo pygame để móc (hook) sự kiện bàn phím
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
                
                # We pump pygame events to catch keystrokes when the debug window is active
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            human_reward = 10.0
                            print(f"[HCRL] Khen thưởng! Timestep {timestep_index}")
                            episode_feedback_count += 1
                        elif event.key == pygame.K_DOWN:
                            human_reward = -10.0
                            print(f"[HCRL] Trừng phạt! Timestep {timestep_index}")
                            episode_feedback_count += 1
                        elif event.key == pygame.K_ESCAPE:
                            raise KeyboardInterrupt

                # Optional: slow down the frame rate so the human has time to react and feedback
                if verbose:
                    time.sleep(0.02) # Thêm xíu độ trễ để người kịp đánh giá

                # Môi trường phản hồi hành động
                observation, step_reward, terminated, _, _ = env.step(action)
                
                # Dựa vào đánh giá của môi trường, cộng thêm đánh giá của con người
                total_reward: Reward = float(step_reward) + human_reward

                # Phạt nếu kết thúc sớm (Quy định gốc của môi trường)
                is_successful = timestep_index >= max_timesteps_per_episode - 1
                if terminated and not is_successful:
                    total_reward = float(-max_episodes_to_run) + human_reward

                # Tác tử học và quyết định bước tiếp theo dựa trên tổng phần thưởng (Môi trường + Con Người)
                action = agent.act(observation, total_reward)

                if terminated or is_successful:
                    print(
                        f"Episode {episode_index} "
                        f"finished after {timestep_index + 1} timesteps. "
                        f"(Human feedbacks provided: {episode_feedback_count})"
                    )

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

        print(f"FAILURE: Goal not reached after {max_episodes_to_run} episodes.")

    except KeyboardInterrupt:
        print("WARNING: Terminated by user request.")
    finally:
        pygame.quit() # Giải phóng sự kiện

    return episode_history


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

    episode_history = run_hcrl_agent(agent=agent, env=env, verbose=verbose)
    save_history(episode_history, experiment_dir="experiment-results")


if __name__ == "__main__":
    main()
