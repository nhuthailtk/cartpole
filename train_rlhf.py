"""
RLHF training for CartPole — Christiano et al. (2017).

"Deep Reinforcement Learning from Human Preferences"
Christiano, Leike, Brown, Martic, Legg, Amodei (NeurIPS 2017)

Pipeline
--------
1. Warm-up: collect segments with an untrained (random-ish) policy.
2. Bootstrap reward model from initial preference labels.
3. Main loop (repeated for NUM_ITERATIONS):
   a. Run episodes using the reward model's predictions as the RL reward.
   b. Collect new trajectory segments.
   c. Sample pairs, query simulated oracle for preferences.
   d. Train reward model on accumulated preferences.
4. Plot learning curves.

Usage
-----
    python train_rlhf.py
"""

import collections
import pathlib
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from cartpole.agents import QLearningAgent
from cartpole.entities import EpisodeHistory, EpisodeHistoryRecord
from cartpole.reward_model import RewardModel, oracle_preference

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

OUTPUT_DIR              = pathlib.Path("experiment-results")

SEED                    = 42

# Segment / preference collection
SEGMENT_LENGTH          = 25    # timesteps per clip shown to oracle
WARMUP_SEGMENTS         = 60    # initial segments collected before RL starts
SEGMENTS_PER_ITER       = 10    # new segments added each iteration
PAIRS_PER_ITER          = 32    # preference queries per reward-model update
REWARD_MODEL_EPOCHS     = 40    # gradient steps per reward-model update
SEGMENT_BUFFER_SIZE     = 500   # max segments kept in replay buffer

# RL training
WARMUP_EPISODES         = 50    # episodes with env reward to kick-start the policy
NUM_ITERATIONS          = 60    # main RLHF iterations
EPISODES_PER_ITER       = 8     # policy episodes per iteration

# Reward model
REWARD_MODEL_LR         = 3e-4
REWARD_MODEL_HIDDEN     = 64

# QLearning agent defaults (same as the existing Q-learning experiments)
AGENT_LR                = 0.2
AGENT_DISCOUNT          = 1.0
AGENT_EXPLORATION       = 0.5
AGENT_EXPLORATION_DECAY = 0.99

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
    """
    Run the agent for exactly `seg_length` steps (resetting mid-segment if
    the episode ends early) and return the sequence of observations as
    (seg_length, obs_dim).

    When `use_reward_model` is not None the agent is trained with the model's
    reward during collection, keeping policy and reward model co-evolving.
    """
    obs_buf: list[np.ndarray] = []
    obs, _ = env.reset()
    action = agent.begin_episode(obs)

    while len(obs_buf) < seg_length:
        obs_buf.append(obs.copy())
        next_obs, env_reward, terminated, truncated, _ = env.step(action)

        # Reward signal for Q-learning update
        if use_reward_model is not None:
            reward = use_reward_model.predict(next_obs)
        else:
            reward = env_reward

        action = agent.act(next_obs, reward)
        obs = next_obs

        if terminated or truncated:
            obs, _ = env.reset()
            action = agent.begin_episode(obs)

    return np.array(obs_buf)   # (seg_length, obs_dim)


def run_episode(
    env: gym.Env,
    agent: QLearningAgent,
    reward_model: RewardModel | None = None,
) -> int:
    """
    Run one full episode.  Returns episode length.

    If `reward_model` is given, uses its predictions as the RL reward
    (RLHF mode).  Otherwise uses the environment reward directly.
    """
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
    """Sample `n_pairs` random pairs from the buffer, query the oracle."""
    segs_a, segs_b, prefs = [], [], []
    indices = list(range(len(segment_buffer)))

    for _ in range(n_pairs):
        i, j = rng.choice(indices, size=2, replace=False)
        seg_a = segment_buffer[i]
        seg_b = segment_buffer[j]
        mu = oracle_preference(seg_a, seg_b, rng)
        segs_a.append(seg_a)
        segs_b.append(seg_b)
        prefs.append(mu)

    return segs_a, segs_b, prefs


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng  = np.random.default_rng(SEED)
    env  = gym.make("CartPole-v1")

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

    segment_buffer: collections.deque[np.ndarray] = collections.deque(
        maxlen=SEGMENT_BUFFER_SIZE
    )

    episode_lengths: list[int] = []
    rm_losses:       list[float] = []

    # ------------------------------------------------------------------ #
    # Phase 1 – Warm-up: train policy on env reward, collect segments     #
    # ------------------------------------------------------------------ #
    print("=== Phase 1: Warm-up ===")
    for ep in range(WARMUP_EPISODES):
        length = run_episode(env, agent, reward_model=None)
        episode_lengths.append(length)
        if (ep + 1) % 10 == 0:
            avg = np.mean(episode_lengths[-10:])
            print(f"  Warmup episode {ep+1:3d}  avg(10)={avg:.1f}")

    # Collect initial segments (policy is somewhat trained now)
    print(f"\nCollecting {WARMUP_SEGMENTS} warm-up segments …")
    for _ in range(WARMUP_SEGMENTS):
        seg = collect_segment(env, agent, SEGMENT_LENGTH, rng, use_reward_model=None)
        segment_buffer.append(seg)

    # Bootstrap reward model
    print(f"Bootstrapping reward model with {PAIRS_PER_ITER * 2} preference pairs …")
    for _ in range(2):                         # two rounds before RL kicks in
        segs_a, segs_b, prefs = sample_preference_pairs(
            list(segment_buffer), PAIRS_PER_ITER, rng
        )
        for _ in range(REWARD_MODEL_EPOCHS):
            loss = reward_model.train_on_preferences(segs_a, segs_b, prefs)
        rm_losses.append(loss)
    print(f"  Bootstrap reward-model loss: {loss:.4f}")

    # ------------------------------------------------------------------ #
    # Phase 2 – RLHF main loop                                            #
    # ------------------------------------------------------------------ #
    print("\n=== Phase 2: RLHF loop ===")
    for iteration in range(1, NUM_ITERATIONS + 1):

        # 2a. Run episodes with reward model as reward signal
        iter_lengths = []
        for _ in range(EPISODES_PER_ITER):
            length = run_episode(env, agent, reward_model=reward_model)
            iter_lengths.append(length)
        episode_lengths.extend(iter_lengths)

        # 2b. Collect new segments (agent updates inside)
        for _ in range(SEGMENTS_PER_ITER):
            seg = collect_segment(
                env, agent, SEGMENT_LENGTH, rng, use_reward_model=reward_model
            )
            segment_buffer.append(seg)

        # 2c. Sample pairs and query oracle
        segs_a, segs_b, prefs = sample_preference_pairs(
            list(segment_buffer), PAIRS_PER_ITER, rng
        )

        # 2d. Train reward model
        for _ in range(REWARD_MODEL_EPOCHS):
            loss = reward_model.train_on_preferences(segs_a, segs_b, prefs)
        rm_losses.append(loss)

        avg_len = np.mean(iter_lengths)
        if iteration % 5 == 0 or iteration == 1:
            print(
                f"  Iter {iteration:3d}/{NUM_ITERATIONS}"
                f"  avg_ep_len={avg_len:6.1f}"
                f"  rm_loss={loss:.4f}"
                f"  buffer={len(segment_buffer)}"
            )

    env.close()

    # ------------------------------------------------------------------ #
    # Save models                                                          #
    # ------------------------------------------------------------------ #
    reward_model.save(OUTPUT_DIR / "reward_model.npz")

    # ------------------------------------------------------------------ #
    # Plotting                                                             #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("RLHF on CartPole — Christiano et al. (2017)", fontsize=13)

    # Episode length curve
    ax = axes[0]
    lengths = np.array(episode_lengths)
    ax.plot(lengths, alpha=0.35, color="steelblue", label="Episode length")
    window = 20
    if len(lengths) >= window:
        rolling = np.convolve(lengths, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(lengths)), rolling, color="steelblue",
                linewidth=2, label=f"Rolling mean ({window})")
    ax.axvline(WARMUP_EPISODES, color="orange", linestyle="--", linewidth=1,
               label="RLHF starts")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Length (timesteps)")
    ax.set_title("Policy performance")
    ax.legend(fontsize=8)
    ax.set_ylim(0, MAX_TIMESTEPS + 10)

    # Reward model loss curve
    ax = axes[1]
    ax.plot(rm_losses, color="tomato", marker="o", markersize=3)
    ax.set_xlabel("Reward model update")
    ax.set_ylabel("Preference cross-entropy loss")
    ax.set_title("Reward model learning")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "rlhf_results.png"
    plt.savefig(plot_path, dpi=120)
    print(f"\nPlot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    train()
