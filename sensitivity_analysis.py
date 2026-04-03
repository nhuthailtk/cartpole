"""
Feedback Weight Sensitivity Analysis
=====================================
Tests how the magnitude of human feedback reward affects Q-Learning.

Inspired by TAMER (Knox & Stone, 2009): the human reward signal's *scale*
relative to the environment reward determines how strongly it shapes the
agent's Q-table. Too small → ignored; too large → overwhelms env signal.

Reference:
    Knox, W. B., & Stone, P. (2009). Interactively shaping agents via human
    reinforcement: The TAMER framework. Proceedings of the 5th International
    Conference on Knowledge Capture (K-CAP), pp. 9-16. ACM.

Methodology:
    - Uses a simulated oracle human (pole angle + cart position heuristic)
      to give *consistent, reproducible* feedback — no human needed.
    - Oracle gives positive feedback when the agent is stable,
      negative feedback when the agent is unstable.
    - Feedback window: Full Feedback (all episodes, 0%-100%).
    - Feedback weights tested: [5, 20, 50]
    - Each condition: 3 random seeds for statistical validity.

Oracle policy (per timestep, 30% trigger probability):
    - Continuous graded feedback in [-weight, +weight]
      proportional to state stability (angle + position).
"""

import argparse
import pathlib
import sys
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cartpole.agents import QLearningAgent
from cartpole.entities import EpisodeHistory, EpisodeHistoryRecord
from cartpole.oracle import oracle_feedback
from cartpole.reward_model import HCRLRewardModel

sys.stdout.reconfigure(encoding="utf-8")

# --- Configuration ---
FEEDBACK_WEIGHTS = [5, 20, 50]
SEEDS = [0, 1, 2]                    # 3 seeds per weight for statistical validity
MAX_TIMESTEPS = 200
TERMINATE_PENALTY = 5000
def get_experiment_dir(max_episodes: int) -> pathlib.Path:
    return pathlib.Path(f"experiment-results/ep{max_episodes}/sensitivity")


# --- Training loop with oracle HCRL ---

def run_oracle_hcrl(weight: float, seed: int, max_episodes: int):
    """
    Train a Q-Learning agent with oracle human feedback at the given reward weight.

    Pipeline (matches train_hcrl.py):
      1. oracle_feedback() fires each step with 50% probability.
      2. If oracle fires  → use signal directly; add to reward model buffer.
      3. If oracle silent → use HCRLRewardModel prediction (once trained).
      4. Fallback to env reward (+1/step) before model has data.
      5. Retrain reward model after each episode.

    Returns (episode_df, agent, reward_model).
    Feedback window: all episodes (Full Feedback).
    """
    rng = np.random.default_rng(seed=seed)
    env = gym.make("CartPole-v1")

    agent = QLearningAgent(
        learning_rate=0.05,
        discount_factor=0.95,
        exploration_rate=0.5,
        exploration_decay_rate=0.99,
        random_state=rng,
    )

    reward_model = HCRLRewardModel(obs_dim=env.observation_space.shape[0],
                                   hidden_dim=64, lr=1e-3)
    model_ready = False
    rm_obs_buf:    list[np.ndarray] = []
    rm_reward_buf: list[float]      = []

    records = []
    for episode_index in range(max_episodes):
        obs, _ = env.reset()
        action = agent.begin_episode(obs)

        for t in range(MAX_TIMESTEPS):
            next_obs, env_reward, terminated, truncated, _ = env.step(action)

            oracle_signal = oracle_feedback(obs, weight, rng)

            if oracle_signal != 0.0:
                rm_obs_buf.append(obs.copy())
                rm_reward_buf.append(oracle_signal)
                shaped = oracle_signal
            elif model_ready:
                shaped = float(reward_model.predict(obs))
            else:
                shaped = env_reward  # fallback before model has data

            is_successful = t >= MAX_TIMESTEPS - 1
            if terminated and not is_successful:
                shaped += float(-TERMINATE_PENALTY)

            action = agent.act(next_obs, shaped)
            obs = next_obs

            if terminated or truncated or is_successful:
                records.append({
                    "episode_index":  episode_index,
                    "episode_length": t + 1,
                    "is_successful":  is_successful,
                })
                break

        # Retrain reward model after each episode
        if len(rm_obs_buf) >= 2:
            reward_model.train_on_feedback(
                np.array(rm_obs_buf), np.array(rm_reward_buf), epochs=20
            )
            model_ready = True

    env.close()
    return pd.DataFrame(records).set_index("episode_index"), agent, reward_model


# --- Run all conditions ---

def run_sensitivity(max_episodes: int):
    """Run all weight × seed combinations and save results."""
    experiment_dir = get_experiment_dir(max_episodes)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    total = len(FEEDBACK_WEIGHTS) * len(SEEDS)
    done = 0

    for weight in FEEDBACK_WEIGHTS:
        for seed in SEEDS:
            done += 1
            print(f"[{done}/{total}] weight={weight:>3}, seed={seed}  ...", end=" ", flush=True)
            df, agent, reward_model = run_oracle_hcrl(weight, seed, max_episodes)
            path = experiment_dir / f"w{weight}_s{seed}.csv"
            df.to_csv(path)
            agent.save(experiment_dir / f"w{weight}_s{seed}_model.npz")
            reward_model.save(experiment_dir / f"w{weight}_s{seed}_reward_model.npz")
            mean = df["episode_length"].mean()
            print(f"mean={mean:.1f}")

    print(f"\nAll results saved to {experiment_dir}/")


# --- Analyze results ---

def analyze_sensitivity(max_episodes: int):
    """Load saved results and produce comparison charts + summary table."""
    experiment_dir = get_experiment_dir(max_episodes)
    results: dict[int, list[pd.DataFrame]] = {w: [] for w in FEEDBACK_WEIGHTS}

    for weight in FEEDBACK_WEIGHTS:
        for seed in SEEDS:
            path = experiment_dir / f"w{weight}_s{seed}.csv"
            if path.exists():
                results[weight].append(pd.read_csv(path, index_col="episode_index"))

    available = [w for w in FEEDBACK_WEIGHTS if results[w]]
    if not available:
        print("No data found. Run without --analyze first.")
        return

    from scipy import stats as scipy_stats

    seeds_str = f"{len(SEEDS)} seed{'s' if len(SEEDS) > 1 else ''}"

    # Print summary table
    col = 14
    table_w = 24 + col * len(available)
    print("\n" + "=" * table_w)
    print(f"  {'FEEDBACK WEIGHT SENSITIVITY (Knox & Stone, 2009)':^{table_w-2}}")
    print(f"  {'Oracle human, Full Feedback (all episodes), ' + seeds_str:^{table_w-2}}")
    print("=" * table_w)
    header = f"  {'Metric':<24}" + "".join(f"weight={w:>2}{'':>{col-8}}" for w in available)
    print(header)
    print("  " + "-" * (table_w - 2))

    wstats: dict[int, dict] = {}
    for w in available:
        dfs = results[w]
        # per-seed last-30 means (for std-across-seeds)
        seed_last30 = [df["episode_length"].tail(30).mean() for df in dfs]
        all_lengths = pd.concat(dfs)["episode_length"]
        last30_all  = pd.concat([df["episode_length"].tail(30) for df in dfs])
        wstats[w] = {
            "mean":        all_lengths.mean(),
            "std":         all_lengths.std(),
            "last30":      last30_all.mean(),
            "last30_std":  np.std(seed_last30) if len(seed_last30) > 1 else 0.0,
            "best":        all_lengths.max(),
            "n_seeds":     len(dfs),
            "all_lengths": all_lengths.values,
        }

    for stat_name, fn in [
        ("Overall Mean ± Std",  lambda w: f"{wstats[w]['mean']:.1f}±{wstats[w]['std']:.1f}"),
        ("Last-30 Avg ± Std",   lambda w: f"{wstats[w]['last30']:.1f}±{wstats[w]['last30_std']:.1f}"),
        ("Best episode",        lambda w: f"{wstats[w]['best']:.0f}"),
        ("N seeds",             lambda w: f"{wstats[w]['n_seeds']}"),
    ]:
        row = f"  {stat_name:<24}" + "".join(f"{fn(w):>{col}}" for w in available)
        print(row)
    print("=" * table_w)

    # Statistical significance: each weight vs weight=1 (lowest) using Mann-Whitney U
    ref_w = available[0]
    print(f"\n  Statistical tests (Mann-Whitney U vs weight={ref_w}, one-sided greater):")
    print(f"  {'Weight':<10} {'Mean':>8} {'U-stat':>10} {'p-value':>10}  {'sig?':>6}")
    print("  " + "-" * 48)
    for w in available:
        if w == ref_w:
            print(f"  w={w:<8} {wstats[w]['mean']:>8.1f} {'(reference)':>22}")
            continue
        u_stat, p_val = scipy_stats.mannwhitneyu(
            wstats[w]["all_lengths"], wstats[ref_w]["all_lengths"], alternative="greater"
        )
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        print(f"  w={w:<8} {wstats[w]['mean']:>8.1f} {u_stat:>10.0f} {p_val:>10.4f}  {sig:>6}")
    print("  (*** p<0.001  ** p<0.01  * p<0.05  ns = not significant)")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Feedback Weight Sensitivity Analysis ({seeds_str})\n"
        "(Oracle human, Full Feedback — all episodes)\n"
        "Knox & Stone (2009) — TAMER Framework",
        fontsize=12, fontweight="bold",
    )

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(available)))

    # 1. Learning curves (mean ± std across seeds)
    ax = axes[0]
    for w, color in zip(available, colors):
        dfs = results[w]
        min_len = min(len(df) for df in dfs)
        stacked = np.stack([df["episode_length"].values[:min_len] for df in dfs])
        mean_curve = pd.Series(stacked.mean(axis=0)).rolling(window=10, min_periods=1).mean()
        std_curve  = pd.Series(stacked.std(axis=0)).rolling(window=10, min_periods=1).mean()
        x = np.arange(min_len)
        ax.plot(x, mean_curve, color=color, linewidth=2, label=f"w={w} (n={len(dfs)})")
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)
    ax.axvspan(0, max_episodes, color="gray", alpha=0.05, label="Feedback window (all episodes)")
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5, label="Goal: 195")
    ax.set_title(f"Learning Curves (mean ± std, {seeds_str})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Overall mean ± std (across all episodes & seeds)
    ax = axes[1]
    means   = [wstats[w]["mean"]     for w in available]
    stds    = [wstats[w]["std"]      for w in available]
    bars = ax.bar([str(w) for w in available], means, color=colors, alpha=0.8,
                  yerr=stds, capsize=5)
    for bar, mv in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"{mv:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5, label="Goal: 195")
    ax.set_title(f"Overall Mean ± Std ({seeds_str})")
    ax.set_xlabel("Feedback Weight")
    ax.set_ylabel("Episode Length")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Last-30 avg ± std across seeds
    ax = axes[2]
    last30s      = [wstats[w]["last30"]     for w in available]
    last30s_std  = [wstats[w]["last30_std"] for w in available]
    bars = ax.bar([str(w) for w in available], last30s, color=colors, alpha=0.8,
                  yerr=last30s_std, capsize=5)
    for bar, mv in zip(bars, last30s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{mv:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5, label="Goal: 195")
    ax.set_title(f"Last-30 Avg ± Std ({seeds_str})\n(End-of-training performance)")
    ax.set_xlabel("Feedback Weight")
    ax.set_ylabel("Episode Length")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = experiment_dir / "sensitivity_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--skip-charts", action="store_true",
                        help="Skip chart generation and plt.show()")
    args = parser.parse_args()

    if args.analyze:
        if not args.skip_charts:
            analyze_sensitivity(args.episodes)
    else:
        run_sensitivity(args.episodes)
        if not args.skip_charts:
            analyze_sensitivity(args.episodes)
