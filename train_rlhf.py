"""
RLHF training for CartPole — Christiano et al. (2017).

"Deep Reinforcement Learning from Human Preferences"
Christiano, Leike, Brown, Martic, Legg, Amodei (NeurIPS 2017)

Pipeline
--------
1. Warm-up (20% of total episodes) with env reward; collect initial segments.
2. Bootstrap reward model from initial oracle preference labels.
3. Main loop for remaining 80% of episodes:
   a. Run EPISODES_PER_ITER episodes using the reward model as reward signal.
   b. Collect new trajectory segments.
   c. Sample pairs, query simulated oracle for preferences.
   d. Train reward model on accumulated preferences.
4. Save agent + history + plot.

Usage
-----
    uv run python train_rlhf.py
    uv run python train_rlhf.py --episodes 200 --seed 0
    uv run python train_rlhf.py --episodes 500 --seed 1
"""

import argparse
import collections
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cartpole.agents import QLearningAgent
from cartpole.reward_model import RewardModel, oracle_preference

# ---------------------------------------------------------------------------
# Fixed hyper-parameters (shared with baseline / HCRL for fair comparison)
# ---------------------------------------------------------------------------

# Agent — identical to run.py and feedback_timing_experiment.py
AGENT_LR                = 0.05
AGENT_DISCOUNT          = 0.95
AGENT_EXPLORATION       = 0.5
AGENT_EXPLORATION_DECAY = 0.99

# Segment / preference collection
SEGMENT_LENGTH          = 25    # timesteps per clip shown to oracle
WARMUP_FRACTION         = 0.20  # fraction of total episodes used as warm-up
EPISODES_PER_ITER       = 8     # policy episodes per RLHF iteration
WARMUP_SEGMENTS         = 40    # initial segments collected before RL starts
SEGMENTS_PER_ITER       = 8     # new segments added each iteration
PAIRS_PER_ITER          = 24    # preference queries per reward-model update
REWARD_MODEL_EPOCHS     = 40    # gradient steps per reward-model update
SEGMENT_BUFFER_SIZE     = 400   # max segments kept in replay buffer

# Reward model
REWARD_MODEL_LR         = 3e-4
REWARD_MODEL_HIDDEN     = 64

MAX_TIMESTEPS           = 200   # CartPole episode cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_segment(
    env: gym.Env,
    agent: QLearningAgent,
    seg_length: int,
    rng: np.random.Generator,
    use_reward_model: RewardModel | None = None,
) -> np.ndarray:
    obs_buf: list[np.ndarray] = []
    obs, _ = env.reset()
    action = agent.begin_episode(obs)

    while len(obs_buf) < seg_length:
        obs_buf.append(obs.copy())
        next_obs, env_reward, terminated, truncated, _ = env.step(action)
        reward = use_reward_model.predict(next_obs) if use_reward_model is not None else env_reward
        action = agent.act(next_obs, reward)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            action = agent.begin_episode(obs)

    return np.array(obs_buf)


def run_episode(
    env: gym.Env,
    agent: QLearningAgent,
    reward_model: RewardModel | None = None,
) -> int:
    obs, _ = env.reset()
    action = agent.begin_episode(obs)
    steps = 0
    while True:
        next_obs, env_reward, terminated, truncated, _ = env.step(action)
        reward = reward_model.predict(next_obs) if reward_model is not None else env_reward
        action = agent.act(next_obs, reward)
        obs = next_obs
        steps += 1
        if terminated or truncated:
            break
    return steps


def sample_preference_pairs(
    segment_buffer: list[np.ndarray],
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    segs_a, segs_b, prefs = [], [], []
    indices = list(range(len(segment_buffer)))
    for _ in range(n_pairs):
        i, j = rng.choice(indices, size=2, replace=False)
        mu = oracle_preference(segment_buffer[i], segment_buffer[j], rng)
        segs_a.append(segment_buffer[i])
        segs_b.append(segment_buffer[j])
        prefs.append(mu)
    return segs_a, segs_b, prefs


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(total_episodes: int, seed: int) -> None:
    warmup_episodes = max(10, int(total_episodes * WARMUP_FRACTION))
    remaining       = total_episodes - warmup_episodes
    num_iterations  = max(1, remaining // EPISODES_PER_ITER)
    actual_total    = warmup_episodes + num_iterations * EPISODES_PER_ITER

    output_dir = pathlib.Path("experiment-results") / f"ep{total_episodes}" / "rlhf-oracle"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  RLHF (oracle)  —  {actual_total} episodes  seed={seed}")
    print(f"  Warm-up: {warmup_episodes} eps  |  Iterations: {num_iterations} × {EPISODES_PER_ITER} eps")
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

    reward_model = RewardModel(
        obs_dim=env.observation_space.shape[0],
        hidden_dim=REWARD_MODEL_HIDDEN,
        lr=REWARD_MODEL_LR,
        rng=rng,
    )

    segment_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=SEGMENT_BUFFER_SIZE)
    episode_lengths: list[int]  = []
    rm_losses:       list[float] = []

    # ------------------------------------------------------------------ #
    # Phase 1 — Warm-up                                                    #
    # ------------------------------------------------------------------ #
    print("\n=== Phase 1: Warm-up ===")
    for ep in range(warmup_episodes):
        length = run_episode(env, agent, reward_model=None)
        episode_lengths.append(length)
        if (ep + 1) % max(1, warmup_episodes // 5) == 0:
            print(f"  ep {ep+1:4d}  avg(last 10)={np.mean(episode_lengths[-10:]):.1f}")

    print(f"\nCollecting {WARMUP_SEGMENTS} warm-up segments…")
    for _ in range(WARMUP_SEGMENTS):
        segment_buffer.append(
            collect_segment(env, agent, SEGMENT_LENGTH, rng, use_reward_model=None)
        )

    print(f"Bootstrapping reward model…")
    for _ in range(2):
        segs_a, segs_b, prefs = sample_preference_pairs(list(segment_buffer), PAIRS_PER_ITER, rng)
        for _ in range(REWARD_MODEL_EPOCHS):
            loss = reward_model.train_on_preferences(segs_a, segs_b, prefs)
        rm_losses.append(loss)
    print(f"  Bootstrap loss: {loss:.4f}")

    # ------------------------------------------------------------------ #
    # Phase 2 — RLHF loop                                                  #
    # ------------------------------------------------------------------ #
    print("\n=== Phase 2: RLHF loop ===")
    for iteration in range(1, num_iterations + 1):
        iter_lengths = []
        for _ in range(EPISODES_PER_ITER):
            iter_lengths.append(run_episode(env, agent, reward_model=reward_model))
        episode_lengths.extend(iter_lengths)

        for _ in range(SEGMENTS_PER_ITER):
            segment_buffer.append(
                collect_segment(env, agent, SEGMENT_LENGTH, rng, use_reward_model=reward_model)
            )

        segs_a, segs_b, prefs = sample_preference_pairs(list(segment_buffer), PAIRS_PER_ITER, rng)
        for _ in range(REWARD_MODEL_EPOCHS):
            loss = reward_model.train_on_preferences(segs_a, segs_b, prefs)
        rm_losses.append(loss)

        if iteration % max(1, num_iterations // 10) == 0 or iteration == 1:
            print(
                f"  iter {iteration:4d}/{num_iterations}"
                f"  avg_ep={np.mean(iter_lengths):6.1f}"
                f"  rm_loss={loss:.4f}"
            )

    env.close()

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    agent.save(output_dir / f"rlhf_oracle_s{seed}_model.npz")
    reward_model.save(output_dir / f"rlhf_oracle_s{seed}_reward_model.npz")
    pd.DataFrame({"episode_length": episode_lengths}).to_csv(
        output_dir / f"rlhf_oracle_s{seed}_history.csv", index_label="episode_index"
    )
    print(f"\nSaved to {output_dir}/")

    # ------------------------------------------------------------------ #
    # Plot                                                                 #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"RLHF (oracle) — {actual_total} episodes, seed={seed}", fontsize=13)

    ax = axes[0]
    lengths = np.array(episode_lengths)
    ax.plot(lengths, alpha=0.3, color="tomato")
    window = 20
    if len(lengths) >= window:
        rolling = np.convolve(lengths, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(lengths)), rolling, color="tomato",
                linewidth=2, label=f"Rolling mean ({window})")
    ax.axvline(warmup_episodes, color="orange", linestyle="--", linewidth=1,
               label="RLHF starts")
    ax.set_xlabel("Episode"); ax.set_ylabel("Length"); ax.set_title("Policy performance")
    ax.legend(fontsize=8); ax.set_ylim(0, MAX_TIMESTEPS + 10); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(rm_losses, color="tomato", marker="o", markersize=3)
    ax.set_xlabel("Reward model update"); ax.set_ylabel("Preference loss")
    ax.set_title("Reward model learning"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"rlhf_oracle_s{seed}_results.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Plot saved to {plot_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLHF training with oracle preferences")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Total training episodes (default: 100)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    args = parser.parse_args()
    train(args.episodes, args.seed)
