"""
Compare RLHF (oracle) vs RLHF (human) vs Baseline.

Produces:
  experiment-results/rlhf_comparison_training.png   — learning curves
  experiment-results/rlhf_comparison_gameplay.png   — gameplay box/bar/hist + stats

Usage
-----
    # After running run.py, train_rlhf.py, and train_rlhf_human.py:
    uv run python compare_rlhf.py

    # Use a different baseline episode count (default: 100):
    uv run python compare_rlhf.py --baseline-episodes 200

    # Change the number of gameplay evaluation episodes:
    uv run python compare_rlhf.py --eval-episodes 200
"""

import argparse
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

from cartpole.agents import QLearningAgent

OUTPUT_DIR = pathlib.Path("experiment-results")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_models(baseline_episodes: int) -> list[dict]:
    base = OUTPUT_DIR / f"ep{baseline_episodes}"
    return [
        {
            "key":          "baseline",
            "label":        "Baseline (Q-Learning)",
            "color":        "steelblue",
            "model_path":   base / "baseline_model.npz",
            # seed-0 canonical history; multi-seed history prefix for averaging
            "history_path": base / "episode_history.csv",
            "history_dir":  base,
            "prefix":       "baseline",
        },
        {
            "key":          "rlhf_oracle",
            "label":        "RLHF (oracle)",
            "color":        "tomato",
            "model_path":   OUTPUT_DIR / "rlhf_model.npz",
            "history_path": OUTPUT_DIR / "rlhf_episode_history.csv",
            "history_dir":  OUTPUT_DIR,
            "prefix":       "rlhf",
        },
        {
            "key":          "rlhf_human",
            "label":        "RLHF (human labels)",
            "color":        "mediumseagreen",
            "model_path":   OUTPUT_DIR / "rlhf_human_model.npz",
            "history_path": OUTPUT_DIR / "rlhf_human_episode_history.csv",
            "history_dir":  OUTPUT_DIR,
            "prefix":       "rlhf_human",
        },
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_history(m: dict) -> pd.DataFrame | None:
    """Return episode_length Series from the model's saved CSV, or None."""
    # Try the canonical single-file path first
    p = pathlib.Path(m["history_path"])
    if p.exists():
        return pd.read_csv(p, index_col=0)

    # Fall back to seed-0 per-seed file
    p0 = pathlib.Path(m["history_dir"]) / f"{m['prefix']}_s0_history.csv"
    if p0.exists():
        return pd.read_csv(p0, index_col=0)

    return None


def evaluate_model(agent: QLearningAgent, num_episodes: int) -> list[int]:
    """Run a trained agent (no exploration, unlimited steps) and return episode lengths."""
    env = gym.make("CartPole-v1", max_episode_steps=None)
    lengths = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        action = agent.begin_episode(obs)
        t = 0
        while True:
            obs, _, terminated, truncated, _ = env.step(action)
            t += 1
            if terminated or truncated:
                lengths.append(t)
                break
            action = agent.act(obs, reward=0.0)
    env.close()
    return lengths


# ---------------------------------------------------------------------------
# Training curve comparison
# ---------------------------------------------------------------------------

def compare_training(models: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Training Curves: Baseline vs RLHF (oracle) vs RLHF (human)",
                 fontsize=13, fontweight="bold")

    window = 15
    any_plotted = False

    for m in models:
        df = _load_history(m)
        if df is None:
            print(f"  [skip] No training history for {m['label']}  ({m['history_path']})")
            continue

        lengths = df["episode_length"].values
        smoothed = pd.Series(lengths).rolling(window=window, min_periods=1).mean().values
        x = np.arange(len(lengths))

        ax.plot(x, smoothed, color=m["color"], linewidth=2.5, label=m["label"])
        ax.fill_between(x, lengths, smoothed, color=m["color"], alpha=0.08)
        any_plotted = True

    if not any_plotted:
        print("  No training histories found — skipping training curve plot.")
        plt.close(fig)
        return

    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5, label="Goal: 195")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode length (timesteps)")
    ax.set_title(f"Rolling mean (window={window})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out = OUTPUT_DIR / "rlhf_comparison_training.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Training chart saved: {out}")


# ---------------------------------------------------------------------------
# Gameplay comparison
# ---------------------------------------------------------------------------

def compare_gameplay(models: list[dict], num_eval_episodes: int) -> None:
    results: list[tuple[dict, list[int]]] = []

    for m in models:
        path = pathlib.Path(m["model_path"])
        if not path.exists():
            print(f"  [skip] Model not found: {path}")
            continue
        print(f"  Evaluating {m['label']} ({num_eval_episodes} episodes)…")
        agent = QLearningAgent.load(path)
        agent.exploration_rate = 0.0
        lengths = evaluate_model(agent, num_eval_episodes)
        results.append((m, lengths))

    if len(results) < 2:
        print("  Need at least 2 models to compare gameplay — skipping.")
        return

    # ------------------------------------------------------------------ #
    # Stats table                                                          #
    # ------------------------------------------------------------------ #
    col_w = 22
    names = [m["label"] for m, _ in results]
    sep = "=" * (26 + col_w * len(results))

    print("\n" + sep)
    print(f"  {'RLHF GAMEPLAY COMPARISON':^{len(sep)-2}}")
    print(sep)
    header = f"  {'Metric':<24}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    print("  " + "-" * (len(sep) - 2))

    for stat_label, fn in [
        ("Mean ± Std",    lambda l: f"{np.mean(l):.1f} ± {np.std(l):.1f}"),
        ("Median",        lambda l: f"{np.median(l):.0f}"),
        ("Min / Max",     lambda l: f"{min(l)} / {max(l)}"),
        ("Rate ≥195 (%)", lambda l: f"{sum(x >= 195 for x in l) / len(l) * 100:.1f}%"),
        ("Rate ≥200 (%)", lambda l: f"{sum(x >= 200 for x in l) / len(l) * 100:.1f}%"),
    ]:
        row = f"  {stat_label:<24}" + "".join(f"{fn(l):>{col_w}}" for _, l in results)
        print(row)
    print(sep)

    # Pairwise significance vs baseline (first in list)
    baseline_m, baseline_l = results[0]
    baseline_arr = np.array(baseline_l, dtype=float)
    bl_mean = np.mean(baseline_arr)
    bl_std  = np.std(baseline_arr, ddof=1)

    print(f"\n  Statistical tests vs {baseline_m['label']} (Welch t-test, two-sided)")
    print(f"  {'Model':<28} {'Mean':>8} {'Δ Mean':>8} {'t':>8} {'p':>10} {'d':>8} {'sig':>6}")
    print("  " + "-" * 82)
    print(f"  {baseline_m['label']:<28} {bl_mean:>8.1f} {'---':>8} {'---':>8} {'---':>10} {'---':>8} {'---':>6}")

    for m, l in results[1:]:
        arr = np.array(l, dtype=float)
        m_mean = np.mean(arr)
        m_std  = np.std(arr, ddof=1)
        delta  = m_mean - bl_mean
        t_stat, p_val = stats.ttest_ind(arr, baseline_arr, equal_var=False)
        pooled = np.sqrt((m_std ** 2 + bl_std ** 2) / 2)
        d = delta / pooled if pooled > 0 else 0.0
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
        print(f"  {m['label']:<28} {m_mean:>8.1f} {delta:>+8.1f} {t_stat:>8.2f} {p_val:>10.4f} {d:>+8.2f} {sig:>6}")

    print("  sig: *** p<0.001  ** p<0.01  * p<0.05  n.s. not significant")
    print("  Cohen's d: |d|<0.2 negligible · <0.5 small · <0.8 medium · ≥0.8 large")

    # ------------------------------------------------------------------ #
    # Gameplay plots                                                        #
    # ------------------------------------------------------------------ #
    all_labels  = [m["label"] for m, _ in results]
    all_colors  = [m["color"] for m, _ in results]
    all_lengths = [l for _, l in results]

    legend_handles = [
        mpatches.Patch(facecolor=c, alpha=0.7, label=lb)
        for lb, c in zip(all_labels, all_colors)
    ]
    goal_line = plt.Line2D([0], [0], color="gray", linestyle="--", alpha=0.7, label="Goal: 195")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Gameplay: Baseline vs RLHF oracle vs RLHF human  ({num_eval_episodes} eval eps)",
        fontsize=13, fontweight="bold",
    )

    # Box plot
    ax = axes[0]
    bp = ax.boxplot(all_lengths, tick_labels=[""] * len(results), patch_artist=True)
    for box, col in zip(bp["boxes"], all_colors):
        box.set_facecolor(col)
        box.set_alpha(0.55)
    ax.axhline(195, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Distribution (box plot)")
    ax.set_ylabel("Episode length (timesteps)")
    ax.legend(handles=legend_handles + [goal_line], fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Bar chart (mean ± std)
    ax = axes[1]
    means = [np.mean(l) for l in all_lengths]
    stds  = [np.std(l)  for l in all_lengths]
    bars  = ax.bar(range(len(results)), means, color=all_colors, alpha=0.75,
                   yerr=stds, capsize=6)
    for bar, mv in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"{mv:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(195, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks([])
    ax.set_title("Mean ± Std")
    ax.set_ylabel("Episode length (timesteps)")
    ax.legend(handles=legend_handles + [goal_line], fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    # Histogram
    ax = axes[2]
    for lengths, col in zip(all_lengths, all_colors):
        ax.hist(lengths, bins=20, alpha=0.4, color=col)
    ax.axvline(195, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Episode length distribution")
    ax.set_xlabel("Episode length (timesteps)")
    ax.set_ylabel("Count")
    ax.legend(handles=legend_handles + [goal_line], fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "rlhf_comparison_gameplay.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Gameplay chart saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Baseline vs RLHF (oracle) vs RLHF (human)"
    )
    parser.add_argument(
        "--baseline-episodes", type=int, default=100,
        help="Episode count used when training the baseline (default: 100)",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=100,
        help="Gameplay evaluation episodes per model (default: 100)",
    )
    args = parser.parse_args()

    models = get_models(args.baseline_episodes)

    print("=" * 60)
    print("  RLHF COMPARISON: Baseline | Oracle | Human")
    print("=" * 60)

    print("\n[1/2] Training curves…")
    compare_training(models)

    print(f"\n[2/2] Gameplay evaluation ({args.eval_episodes} episodes each)…")
    compare_gameplay(models, args.eval_episodes)

    plt.show()


if __name__ == "__main__":
    main()
