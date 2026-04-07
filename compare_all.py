"""
Compare ALL training methods:
  Baseline | HCRL Early/Mid/Late/Full | RLHF oracle | RLHF human

Produces:
  experiment-results/compare_all_training.png   — learning curves (all methods)
  experiment-results/compare_all_gameplay.png   — gameplay box / bar / stats

Usage
-----
    uv run python compare_all.py
    uv run python compare_all.py --episodes 200 --eval-episodes 100
    uv run python compare_all.py --episodes 200 --skip-missing
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

from cartpole import config as cfg
from cartpole.agents import QLearningAgent

OUTPUT_DIR = pathlib.Path("experiment-results")

SEEDS = cfg.SEEDS


# ---------------------------------------------------------------------------
# Model registry — all methods
# ---------------------------------------------------------------------------



def get_all_models(episodes: int, seed: int) -> list[dict]:
    base         = OUTPUT_DIR / f"ep{episodes}"
    timing       = base / "timing-experiment"
    oracle       = base / "rlhf-oracle"
    human        = base / "rlhf-human"
    hcrl_oracle  = base / "hcrl-oracle"
    hcrl_human   = base / "hcrl-human"

    return [
        # ── Baseline ──────────────────────────────────────────────────────
        {
            "key":         "baseline",
            "label":       "Baseline",
            "color":       "steelblue",
            "linestyle":   "--",
            "group":       "Baseline",
            "model_path":  base / f"baseline_s{seed}_model.npz",
            "history_dir": base,
            "prefix":      "baseline",
            "seed":        seed,
        },
        # ── HCRL timing conditions ────────────────────────────────────────
        {
            "key":         "hcrl_early",
            "label":       "HCRL Early (0-20%)",
            "color":       "limegreen",
            "linestyle":   "-",
            "group":       "HCRL",
            "model_path":  timing / f"early_s{seed}_model.npz",
            "history_dir": timing,
            "prefix":      "early",
            "seed":        seed,
        },
        {
            "key":         "hcrl_mid",
            "label":       "HCRL Mid (40-60%)",
            "color":       "orange",
            "linestyle":   "-",
            "group":       "HCRL",
            "model_path":  timing / f"mid_s{seed}_model.npz",
            "history_dir": timing,
            "prefix":      "mid",
            "seed":        seed,
        },
        {
            "key":         "hcrl_late",
            "label":       "HCRL Late (80-100%)",
            "color":       "darkorchid",
            "linestyle":   "-",
            "group":       "HCRL",
            "model_path":  timing / f"late_s{seed}_model.npz",
            "history_dir": timing,
            "prefix":      "late",
            "seed":        seed,
        },
        {
            "key":         "hcrl_full",
            "label":       "HCRL Full Feedback",
            "color":       "crimson",
            "linestyle":   "-",
            "group":       "HCRL",
            "model_path":  timing / f"full_feedback_s{seed}_model.npz",
            "history_dir": timing,
            "prefix":      "full_feedback",
            "seed":        seed,
        },
        # ── RLHF ──────────────────────────────────────────────────────────
        {
            "key":         "rlhf_oracle",
            "label":       "RLHF (oracle)",
            "color":       "tomato",
            "linestyle":   "-.",
            "group":       "RLHF",
            "model_path":  oracle / f"rlhf_oracle_s{seed}_model.npz",
            "history_dir": oracle,
            "prefix":      f"rlhf_oracle_s{seed}",
            "seed":        seed,
        },
        {
            "key":         "rlhf_human",
            "label":       "RLHF (human)",
            "color":       "mediumseagreen",
            "linestyle":   "-.",
            "group":       "RLHF",
            "model_path":  human / f"rlhf_human_s{seed}_model.npz",
            "history_dir": human,
            "prefix":      f"rlhf_human_s{seed}",
            "seed":        seed,
        },
        # ── HCRL oracle + human ───────────────────────────────────────────
        {
            "key":         "hcrl_oracle",
            "label":       "HCRL (oracle)",
            "color":       "forestgreen",
            "linestyle":   "-",
            "group":       "HCRL",
            "model_path":  hcrl_oracle / f"hcrl_oracle_s{seed}_model.npz",
            "history_dir": hcrl_oracle,
            "prefix":      f"hcrl_oracle_s{seed}",
            "seed":        seed,
        },
        {
            "key":         "hcrl_human",
            "label":       "HCRL (human)",
            "color":       "deepskyblue",
            "linestyle":   "-",
            "group":       "HCRL",
            "model_path":  hcrl_human / f"hcrl_human_s{seed}_model.npz",
            "history_dir": hcrl_human,
            "prefix":      f"hcrl_human_s{seed}",
            "seed":        seed,
        },
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_histories(m: dict) -> list[pd.DataFrame]:
    """Return list of episode-length DataFrames for this model's seed."""
    seed = m["seed"]
    candidates = [
        pathlib.Path(m["history_dir"]) / f"{m['prefix']}_history.csv",
        pathlib.Path(m["history_dir"]) / f"{m['prefix']}_s{seed}_history.csv",
        # Fallback: old canonical single-seed files
        pathlib.Path(m["history_dir"]) / f"{m['prefix']}_episode_history.csv",
        pathlib.Path(m["history_dir"]) / "episode_history.csv",
    ]
    for p in candidates:
        if p.exists():
            return [pd.read_csv(p, index_col=0)]
    return []


def evaluate_model(agent: QLearningAgent, num_episodes: int) -> list[int]:
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
# 1. Training curves
# ---------------------------------------------------------------------------

def plot_training(models: list[dict], episodes: int) -> None:
    WINDOW = 15

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(
        f"Learning Curves — All Methods  ({episodes} training eps, rolling mean {WINDOW})",
        fontsize=13, fontweight="bold",
    )

    # Left: all methods together
    ax_all = axes[0]
    ax_all.set_title("All methods")

    # Right: grouped mean per paradigm
    ax_grp = axes[1]
    ax_grp.set_title("By paradigm (mean across conditions)")

    group_curves: dict[str, list[np.ndarray]] = {"Baseline": [], "HCRL": [], "RLHF": []}
    group_colors  = {"Baseline": "steelblue", "HCRL": "darkorange", "RLHF": "mediumseagreen"}

    any_plotted = False
    for m in models:
        dfs = _load_histories(m)
        if not dfs:
            print(f"  [skip] No history: {m['label']}")
            continue

        min_len = min(len(df) for df in dfs)
        stacked = np.stack([df["episode_length"].values[:min_len] for df in dfs])
        mean_c  = pd.Series(stacked.mean(axis=0)).rolling(WINDOW, min_periods=1).mean().values
        std_c   = pd.Series(stacked.std(axis=0)).rolling(WINDOW, min_periods=1).mean().values
        x = np.arange(min_len)

        ax_all.plot(x, mean_c, color=m["color"], linewidth=2,
                    linestyle=m["linestyle"], label=m["label"])
        ax_all.fill_between(x, mean_c - std_c, mean_c + std_c,
                            color=m["color"], alpha=0.10)

        group_curves[m["group"]].append(mean_c)
        any_plotted = True

    if not any_plotted:
        print("  No training histories found.")
        plt.close(fig)
        return

    for group, curves in group_curves.items():
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        stacked = np.stack([c[:min_len] for c in curves])
        mean_g  = stacked.mean(axis=0)
        std_g   = stacked.std(axis=0)
        x = np.arange(min_len)
        ax_grp.plot(x, mean_g, color=group_colors[group], linewidth=2.5, label=group)
        ax_grp.fill_between(x, mean_g - std_g, mean_g + std_g,
                            color=group_colors[group], alpha=0.15)

    for ax in axes:
        ax.axhline(195, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode length (timesteps)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 215)

    # Left panel: shared legend below (too many lines to fit inside)
    handles_all, labels_all = axes[0].get_legend_handles_labels()
    goal_patch = plt.Line2D([0], [0], color="gray", linestyle="--", alpha=0.7, label="Goal: 195")

    fig.legend(
        handles=handles_all + [goal_patch],
        loc="lower center",
        ncols=5,
        fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.18),
    )

    # Right panel (paradigm groups): legend inside — only 4 items, fits fine
    axes[1].legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    out = OUTPUT_DIR / f"ep{episodes}" / "compare_all_training.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Training chart saved: {out}")


# ---------------------------------------------------------------------------
# 2. Gameplay comparison
# ---------------------------------------------------------------------------

def plot_gameplay(models: list[dict], num_eval: int, episodes: int = 100) -> None:
    results: list[tuple[dict, list[int]]] = []

    for m in models:
        path = pathlib.Path(m["model_path"])
        if not path.exists():
            print(f"  [skip] Model not found: {path}")
            continue
        print(f"  Evaluating {m['label']}…")
        agent = QLearningAgent.load(path)
        agent.exploration_rate = 0.0
        lengths = evaluate_model(agent, num_eval)
        results.append((m, lengths))

    if len(results) < 2:
        print("  Need at least 2 models — skipping gameplay plot.")
        return

    # ── Stats table ────────────────────────────────────────────────────── #
    col_w = 20
    labels = [m["label"] for m, _ in results]
    sep = "=" * (26 + col_w * len(results))

    print("\n" + sep)
    print(f"  {'GAMEPLAY COMPARISON — ALL METHODS':^{len(sep)-2}}")
    print(sep)
    print(f"  {'Metric':<24}" + "".join(f"{lb:>{col_w}}" for lb in labels))
    print("  " + "-" * (len(sep) - 2))

    for stat_label, fn in [
        ("Mean ± Std",    lambda l: f"{np.mean(l):.1f} ± {np.std(l):.1f}"),
        ("Median",        lambda l: f"{np.median(l):.0f}"),
        ("Min / Max",     lambda l: f"{min(l)} / {max(l)}"),
        ("Rate ≥195 (%)", lambda l: f"{sum(x >= 195 for x in l)/len(l)*100:.1f}%"),
        ("Rate ≥200 (%)", lambda l: f"{sum(x >= 200 for x in l)/len(l)*100:.1f}%"),
    ]:
        print(f"  {stat_label:<24}" + "".join(f"{fn(l):>{col_w}}" for _, l in results))
    print(sep)

    # ── Significance vs baseline ───────────────────────────────────────── #
    bl_m, bl_l = results[0]
    bl_arr     = np.array(bl_l, dtype=float)
    bl_mean    = np.mean(bl_arr)
    bl_std     = np.std(bl_arr, ddof=1)

    print(f"\n  Statistical tests vs {bl_m['label']} (Welch t-test, two-sided)")
    print(f"  {'Model':<30} {'Mean':>8} {'Δ':>7} {'t':>8} {'p':>10} {'d':>8} {'sig':>6}")
    print("  " + "-" * 82)
    print(f"  {bl_m['label']:<30} {bl_mean:>8.1f} {'—':>7} {'—':>8} {'—':>10} {'—':>8} {'ref':>6}")

    for m, l in results[1:]:
        arr    = np.array(l, dtype=float)
        m_mean = np.mean(arr)
        m_std  = np.std(arr, ddof=1)
        delta  = m_mean - bl_mean
        t_stat, p_val = stats.ttest_ind(arr, bl_arr, equal_var=False)
        pooled = np.sqrt((m_std ** 2 + bl_std ** 2) / 2)
        d      = delta / pooled if pooled > 0 else 0.0
        sig    = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
        print(f"  {m['label']:<30} {m_mean:>8.1f} {delta:>+7.1f} {t_stat:>8.2f} {p_val:>10.4f} {d:>+8.2f} {sig:>6}")

    print("  sig: *** p<0.001  ** p<0.01  * p<0.05  n.s. not significant")
    print("  Cohen's d: |d|<0.2 negligible · <0.5 small · <0.8 medium · ≥0.8 large")

    # ── Plots ─────────────────────────────────────────────────────────── #
    all_labels  = [m["label"] for m, _ in results]
    all_colors  = [m["color"] for m, _ in results]
    all_lengths = [l for _, l in results]

    legend_handles = [
        mpatches.Patch(facecolor=c, alpha=0.75, label=lb)
        for lb, c in zip(all_labels, all_colors)
    ]
    goal_line = plt.Line2D([0], [0], color="gray", linestyle="--", alpha=0.7, label="Goal: 195")
    all_handles = legend_handles + [goal_line]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(
        f"Gameplay Comparison — All Methods  ({num_eval} eval episodes each)",
        fontsize=13, fontweight="bold",
    )

    # Box plot
    ax = axes[0]
    bp = ax.boxplot(all_lengths, tick_labels=[""] * len(results), patch_artist=True)
    for box, col in zip(bp["boxes"], all_colors):
        box.set_facecolor(col); box.set_alpha(0.55)
    ax.axhline(195, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Distribution")
    ax.set_ylabel("Episode length (timesteps)")
    ax.grid(True, alpha=0.3)

    # Bar chart
    ax = axes[1]
    means = [np.mean(l) for l in all_lengths]
    stds  = [np.std(l)  for l in all_lengths]
    bars  = ax.bar(range(len(results)), means, color=all_colors, alpha=0.75,
                   yerr=stds, capsize=5)
    for bar, mv in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{mv:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(195, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks([])
    ax.set_title("Mean ± Std")
    ax.set_ylabel("Episode length (timesteps)")
    ax.grid(True, alpha=0.3, axis="y")

    # Histogram
    ax = axes[2]
    for lengths, col in zip(all_lengths, all_colors):
        ax.hist(lengths, bins=20, alpha=0.35, color=col)
    ax.axvline(195, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Distribution histogram")
    ax.set_xlabel("Episode length (timesteps)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    # Shared legend below all panels — avoids overlapping data with 10+ models
    fig.legend(
        handles=all_handles,
        loc="lower center",
        ncols=6,
        fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.12),
    )

    plt.tight_layout()
    out = OUTPUT_DIR / f"ep{episodes}" / "compare_all_gameplay.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Gameplay chart saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Baseline, HCRL (4 conditions), RLHF oracle, RLHF human"
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Training episodes used for ALL methods (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Which seed to load for each method (default: 0)",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=100,
        help="Gameplay evaluation episodes per model (default: 100)",
    )
    args = parser.parse_args()

    models = get_all_models(args.episodes, args.seed)

    print("=" * 70)
    print(f"  FULL COMPARISON: Baseline | HCRL ×4 | RLHF ×2")
    print(f"  episodes={args.episodes}  seed={args.seed}  eval={args.eval_episodes}")
    print("=" * 70)

    print(f"\n[1/2] Training curves…")
    plot_training(models, args.episodes)

    print(f"\n[2/2] Gameplay evaluation  ({args.eval_episodes} eps each)…")
    plot_gameplay(models, args.eval_episodes, args.episodes)

    plt.show()


if __name__ == "__main__":
    main()
