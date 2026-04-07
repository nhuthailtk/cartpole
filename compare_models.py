import argparse
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

from cartpole import config as cfg
from cartpole.agents import QLearningAgent

SEEDS = cfg.SEEDS


def get_models(episodes: int) -> list[dict]:
    base = pathlib.Path(f"experiment-results/ep{episodes}")
    timing = base / "timing-experiment"
    return [
        {
            "name": "baseline_shaped",
            "label": "Baseline",
            "color": "blue",
            "model_path": str(base / "baseline_model.npz"),
            "history_path": str(base / "episode_history.csv"),
            "history_dir": base,
            "prefix": "baseline",
        },
        {
            "name": "early",
            "label": "HCRL Early (0-20%)",
            "color": "green",
            "model_path": str(timing / "early_model.npz"),
            "history_path": str(timing / "early_episode_history.csv"),
            "history_dir": timing,
            "prefix": "early",
        },
        {
            "name": "mid",
            "label": "HCRL Mid (40-60%)",
            "color": "orange",
            "model_path": str(timing / "mid_model.npz"),
            "history_path": str(timing / "mid_episode_history.csv"),
            "history_dir": timing,
            "prefix": "mid",
        },
        {
            "name": "late",
            "label": "HCRL Late (80-100%)",
            "color": "purple",
            "model_path": str(timing / "late_model.npz"),
            "history_path": str(timing / "late_episode_history.csv"),
            "history_dir": timing,
            "prefix": "late",
        },
        {
            "name": "full_feedback",
            "label": "HCRL Full Feedback",
            "color": "red",
            "model_path": str(timing / "full_feedback_model.npz"),
            "history_path": str(timing / "full_feedback_episode_history.csv"),
            "history_dir": timing,
            "prefix": "full_feedback",
        },
    ]


def load_seed_histories(m: dict) -> list[pd.DataFrame]:
    """Load per-seed CSVs; fall back to canonical single-seed file."""
    dfs = []
    for seed in SEEDS:
        p = pathlib.Path(m["history_dir"]) / f"{m['prefix']}_s{seed}_history.csv"
        if p.exists():
            dfs.append(pd.read_csv(p, index_col="episode_index"))
    if not dfs and pathlib.Path(m["history_path"]).exists():
        dfs.append(pd.read_csv(m["history_path"], index_col="episode_index"))
    return dfs


def evaluate_model(agent: QLearningAgent, num_episodes: int = 100) -> list[int]:
    """Run a trained agent for num_episodes with unlimited steps and return the list of episode lengths."""
    env = gym.make("CartPole-v1", max_episode_steps=None)
    episode_lengths: list[int] = []

    for _ in range(num_episodes):
        observation, _ = env.reset()
        action = agent.begin_episode(observation)
        timestep = 0

        while True:
            observation, _, terminated, truncated, _ = env.step(action)
            timestep += 1
            if terminated or truncated:
                episode_lengths.append(timestep)
                break
            action = agent.act(observation, reward=0.0)

    env.close()
    return episode_lengths


def compare_training_curves(episodes: int = 100) -> bool:
    """Plot training learning curves (mean ± std across seeds) for all models."""
    MODELS = get_models(episodes)
    out_dir = pathlib.Path(f"experiment-results/ep{episodes}")

    loaded = {}
    for m in MODELS:
        dfs = load_seed_histories(m)
        if dfs:
            loaded[m["name"]] = (m, dfs)
        else:
            print(f"  Data not found: {m['history_path']}")

    if not loaded:
        print("No training data found.")
        return False

    seeds_str = f"{len(SEEDS)} seeds"
    window_size = 10
    plt.figure(figsize=(14, 6))

    for name, (m, dfs) in loaded.items():
        n_seeds = len(dfs)
        min_len = min(len(df) for df in dfs)
        stacked = np.stack([df["episode_length"].values[:min_len] for df in dfs])
        mean_curve = pd.Series(stacked.mean(axis=0)).rolling(window=window_size, min_periods=1).mean()
        std_curve  = pd.Series(stacked.std(axis=0)).rolling(window=window_size, min_periods=1).mean()
        x = np.arange(min_len)
        plt.plot(x, mean_curve, label=f"{m['label']} (n={n_seeds})",
                 color=m["color"], linewidth=2)
        plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                         color=m["color"], alpha=0.12)

    plt.title(f"Training Curves: Baseline vs HCRL ({episodes} eps, mean ± std, {seeds_str})")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.axhline(y=195, color="gray", linestyle="--", alpha=0.5, label="Goal: 195")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = out_dir / "comparison_training.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Training chart saved to: {output_path}")
    return True


def compare_gameplay(episodes: int = 100, num_episodes: int = 100) -> bool:
    """Load all 4 saved models and compare their gameplay performance."""
    MODELS = get_models(episodes)
    out_dir = pathlib.Path(f"experiment-results/ep{episodes}")
    results: dict[str, tuple[dict, list[int]]] = {}

    for m in MODELS:
        path = pathlib.Path(m["model_path"])
        if not path.exists():
            print(f"  Model not found: {path}")
            continue
        print(f"  Evaluating {m['label']} ({num_episodes} episodes)...")
        agent = QLearningAgent.load(path)
        lengths = evaluate_model(agent, num_episodes)
        results[m["name"]] = (m, lengths)

    if len(results) < 2:
        print("Need at least 2 models to compare.")
        return False

    # Print statistics table
    names = list(results.keys())
    col_width = 18

    print("\n" + "=" * (25 + col_width * len(names)))
    header = f"{'Metric':<25}"
    for n in names:
        header += f"{results[n][0]['label']:>{col_width}}"
    print(header)
    print("=" * (25 + col_width * len(names)))

    for stat_name, stat_fn in [
        ("Mean \u00b1 Std",      lambda l: f"{np.mean(l):.1f} \u00b1 {np.std(l):.1f}"),
        ("Median",           lambda l: f"{np.median(l):.1f}"),
        ("Min / Max",        lambda l: f"{min(l)} / {max(l)}"),
        ("Episodes \u2265200",    lambda l: f"{sum(1 for x in l if x >= 200)}"),
        ("Rate \u2265195 (%)",    lambda l: f"{sum(1 for x in l if x >= 195) / len(l) * 100:.1f}%"),
    ]:
        row = f"{stat_name:<25}"
        for n in names:
            row += f"{stat_fn(results[n][1]):>{col_width}}"
        print(row)
    print("=" * (25 + col_width * len(names)))

    # --- Statistical evaluation table (vs Baseline) ---
    baseline_key = "baseline_shaped"
    if baseline_key in results:
        baseline_lengths = np.array(results[baseline_key][1], dtype=float)
        hcrl_names = [n for n in names if n != baseline_key]

        if hcrl_names:
            print("\n" + "=" * 90)
            print(f"  {'STATISTICAL EVALUATION vs BASELINE (Welch t-test, two-sided)':^88}")
            print("=" * 90)
            print(f"  {'Condition':<22}{'Mean \u00b1 Std':>16}{'\u0394 Mean':>10}{'t-stat':>10}{'p-value':>12}{'Cohen d':>10}{'Sig.':>8}")
            print("  " + "-" * 88)

            bl_mean, bl_std = np.mean(baseline_lengths), np.std(baseline_lengths, ddof=1)
            print(f"  {'Baseline (ref.)':<22}{bl_mean:>8.1f} \u00b1 {bl_std:<6.1f}{'---':>10}{'---':>10}{'---':>12}{'---':>10}{'---':>8}")

            for n in hcrl_names:
                m_info, lengths = results[n]
                arr = np.array(lengths, dtype=float)
                m_mean, m_std = np.mean(arr), np.std(arr, ddof=1)
                delta = m_mean - bl_mean

                # Welch's t-test (does not assume equal variance)
                t_stat, p_val = stats.ttest_ind(arr, baseline_lengths, equal_var=False)

                # Cohen's d (pooled std)
                pooled_std = np.sqrt((m_std**2 + bl_std**2) / 2)
                cohens_d = delta / pooled_std if pooled_std > 0 else 0.0

                # Significance markers
                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"
                else:
                    sig = "n.s."

                # Effect size interpretation
                abs_d = abs(cohens_d)
                if abs_d < 0.2:
                    eff = "negligible"
                elif abs_d < 0.5:
                    eff = "small"
                elif abs_d < 0.8:
                    eff = "medium"
                else:
                    eff = "large"

                print(f"  {m_info['label']:<22}{m_mean:>8.1f} \u00b1 {m_std:<6.1f}{delta:>+10.1f}{t_stat:>10.2f}{p_val:>12.4f}{cohens_d:>+10.2f}{sig:>8}")

            print("=" * 90)
            print("  Sig.: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant")
            print(f"  Cohen's d: |d|<0.2 negligible, <0.5 small, <0.8 medium, \u22650.8 large")
            print(f"  Evaluation: {num_episodes} gameplay episodes per model, unlimited steps")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Gameplay Comparison: Baseline vs HCRL ({episodes} training eps)", fontsize=13, fontweight="bold")

    all_labels = [results[n][0]["label"] for n in names]
    all_colors = [results[n][0]["color"] for n in names]
    all_lengths = [results[n][1] for n in names]

    import matplotlib.patches as mpatches

    # Reusable legend handles (colored patch per model + goal line)
    legend_handles = [
        mpatches.Patch(facecolor=c, alpha=0.7, label=l)
        for l, c in zip(all_labels, all_colors)
    ]
    goal_handle = plt.Line2D([0], [0], color="gray", linestyle="--", alpha=0.7, label="Goal: 195")

    # 1. Box plot — legend inside upper right
    ax = axes[0]
    bp = ax.boxplot(all_lengths, tick_labels=[""] * len(names), patch_artist=True)
    for box, color in zip(bp["boxes"], all_colors):
        box.set_facecolor(color)
        box.set_alpha(0.5)
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Box Plot")
    ax.set_ylabel("Episode Length")
    ax.legend(handles=legend_handles + [goal_handle], fontsize=7.5,
              loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # 2. Bar chart (mean ± std) — legend inside upper left
    ax = axes[1]
    means = [np.mean(l) for l in all_lengths]
    stds  = [np.std(l)  for l in all_lengths]
    bars = ax.bar(range(len(names)), means, color=all_colors, alpha=0.7, yerr=stds, capsize=5)
    for bar, mv in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{mv:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks([])
    ax.set_title("Mean Episode Length ± Std")
    ax.set_ylabel("Episode Length")
    ax.legend(handles=legend_handles + [goal_handle], fontsize=7.5,
              loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Histogram — legend inside upper right
    ax = axes[2]
    for lengths, color in zip(all_lengths, all_colors):
        ax.hist(lengths, bins=20, alpha=0.4, color=color)
    ax.axvline(x=195, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Episode Length Distribution")
    ax.set_xlabel("Episode Length")
    ax.set_ylabel("Count")
    ax.legend(handles=legend_handles + [goal_handle], fontsize=7.5,
              loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = out_dir / "comparison_gameplay.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Gameplay chart saved to: {output_path}")
    return True


def compare_models(episodes: int = 100, num_eval_episodes: int = 100) -> None:
    print("=" * 60)
    print(f" COMPARING 5 MODELS: 1 Baseline + 4 HCRL ({episodes} training eps)")
    print("=" * 60)

    print("\n[1/2] Comparing training curves...")
    compare_training_curves(episodes)

    print(f"\n[2/2] Comparing gameplay performance ({num_eval_episodes} episodes per model)...")
    compare_gameplay(episodes, num_eval_episodes)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=100)
    args = parser.parse_args()
    compare_models(args.episodes, args.eval_episodes)
