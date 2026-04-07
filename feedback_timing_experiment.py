"""
Feedback Timing Experiment
==========================
Tests when human feedback is most effective during HCRL training.

4 conditions:
  - Early:         feedback in episodes 0–20%
  - Mid:           feedback in episodes 40–60%
  - Late:          feedback in episodes 80–100%
  - Full Feedback: feedback throughout all episodes (no window)

After all 4 runs, compares results with charts.

Research question inspired by TAMER (Knox & Stone, 2009), which established
that human reward signals are most useful when the agent has a stable policy
to correct. This experiment directly tests that hypothesis across training phases.

Reference:
    Knox, W. B., & Stone, P. (2009). Interactively shaping agents via human
    reinforcement: The TAMER framework. Proceedings of the 5th International
    Conference on Knowledge Capture (K-CAP), pp. 9-16. ACM.
    https://doi.org/10.1145/1597735.1597738

Usage:
    # Automated (oracle human, no keyboard needed):
    uv run python feedback_timing_experiment.py --auto --episodes 200
    uv run python feedback_timing_experiment.py --auto --episodes 500

    # Analyze only (after training):
    uv run python feedback_timing_experiment.py --analyze --episodes 200

    # Human-in-the-loop (interactive):
    uv run python feedback_timing_experiment.py --episodes 200
"""

import argparse
import pathlib

import gymnasium as gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cartpole import config as cfg
from cartpole.agents import QLearningAgent
from cartpole.entities import EpisodeHistory, EpisodeHistoryRecord
from cartpole.reward_model import HCRLRewardModel
from cartpole.train_utils import (
    make_agent,
    make_episode_history,
    run_hcrl_episode,
    save_episode_history_csv,
    save_feedback_csv,
)
from train_hcrl import run_hcrl_agent

COLORS = ["green", "orange", "purple", "red"]
SEEDS  = cfg.SEEDS


def get_conditions(max_episodes: int) -> list[dict]:
    """Build the 4 timing conditions scaled to max_episodes."""
    n = max_episodes
    return [
        {"name": k, "label": f"{k.replace('_', ' ').title()} ({int(v[0]*100)}-{int(v[1]*100)}%)",
         "window": (int(v[0] * n), int(v[1] * n))}
        for k, v in cfg.TIMING_CONDITIONS.items()
    ]


def run_oracle_condition(
    name: str,
    window: tuple[int, int],
    max_episodes: int,
    seed: int,
    feedback_weight: float = cfg.HCRL_FEEDBACK_WEIGHT,
):
    """
    Train one timing condition using run_hcrl_episode() from train_utils.

    The oracle fires only when the current episode is inside [fb_start, fb_end).
    The reward model is retrained after every episode on all collected signals.

    Returns (episode_history, feedback_log, agent, reward_model).
    """
    rng = np.random.default_rng(seed=seed)
    env = gym.make("CartPole-v1")
    fb_start, fb_end = window

    agent        = make_agent(rng)
    reward_model = HCRLRewardModel(obs_dim=env.observation_space.shape[0])
    model_ready  = False

    episode_history = make_episode_history()
    all_feedback:  list[dict]       = []
    rm_obs_buf:    list[np.ndarray] = []
    rm_reward_buf: list[float]      = []

    for ep in range(max_episodes):
        in_window = fb_start <= ep < fb_end
        ep_len, new_obs, new_rew, fb_log = run_hcrl_episode(
            env, agent, reward_model if model_ready else None, rng,
            feedback_weight=feedback_weight,
            in_feedback_window=in_window,
        )
        # Annotate feedback log with episode index
        for entry in fb_log:
            entry["episode"] = ep
            entry["timestep"] = entry.pop("episode_step", 0)
        all_feedback.extend(fb_log)
        rm_obs_buf.extend(new_obs)
        rm_reward_buf.extend(new_rew)

        episode_history.record_episode(EpisodeHistoryRecord(
            episode_index=ep, episode_length=ep_len,
            is_successful=ep_len >= cfg.MAX_TIMESTEPS,
        ))

        if len(rm_obs_buf) >= 2:
            reward_model.train_on_feedback(np.array(rm_obs_buf), np.array(rm_reward_buf))
            model_ready = True

    env.close()
    print(f"    {name}: {max_episodes} eps done, feedback={len(all_feedback)}")
    return episode_history, all_feedback, agent, reward_model


# Backward-compatible wrappers (used by run_experiment below)
def save_feedback_log(feedback_log, experiment_dir, filename="hcrl_feedback_log.csv"):
    return save_feedback_csv(feedback_log, pathlib.Path(experiment_dir) / filename)


def save_history(history, experiment_dir, filename="hcrl_episode_history.csv"):
    return save_episode_history_csv(history, pathlib.Path(experiment_dir) / filename)


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

def run_experiment(max_episodes: int, auto: bool, experiment_dir: pathlib.Path) -> None:
    """Run all 4 timing conditions × 3 seeds (oracle or human)."""
    experiment_dir.mkdir(parents=True, exist_ok=True)
    conditions = get_conditions(max_episodes)
    total = len(conditions) * len(SEEDS)
    done = 0

    for i, cond in enumerate(conditions):
        name = cond["name"]
        window = cond["window"]
        label = cond["label"]

        print("\n" + "#" * 60)
        print(f"  CONDITION {i + 1}/{len(conditions)}: {label}")
        print(f"  Feedback window: Episode {window[0]} → {window[1]}")
        print(f"  Total episodes: {max_episodes}  |  Seeds: {SEEDS}")
        print(f"  Mode: {'Oracle (automated)' if auto else 'Human (interactive)'}")
        print("#" * 60)

        for seed in SEEDS:
            done += 1
            print(f"  [{done}/{total}] seed={seed} ...", end=" ", flush=True)

            if auto:
                history, feedback_log, agent, reward_model = run_oracle_condition(
                    name=name, window=window, max_episodes=max_episodes, seed=seed,
                )
                save_history(history, str(experiment_dir),
                             filename=f"{name}_s{seed}_history.csv")
                save_feedback_log(feedback_log, str(experiment_dir),
                                  filename=f"{name}_s{seed}_feedback_log.csv")
                agent.save(str(experiment_dir / f"{name}_s{seed}_model.npz"))
                reward_model.save(str(experiment_dir / f"{name}_s{seed}_reward_model.npz"))
                lengths = [r.episode_length for r in history.all_records()]
                print(f"mean={np.mean(lengths):.1f}, last-30={np.mean(lengths[-30:]):.1f}")
            else:
                # Interactive: human can only run once — use seed=0 only
                if seed != 0:
                    print("skipped (interactive mode uses seed=0 only)")
                    continue
                random_state = np.random.default_rng(seed=0)
                env = gym.make("CartPole-v1", render_mode="human")
                agent = QLearningAgent(
                    learning_rate=0.05,
                    discount_factor=0.95,
                    exploration_rate=0.5,
                    exploration_decay_rate=0.99,
                    random_state=random_state,
                )
                episode_history, feedback_log = run_hcrl_agent(
                    agent=agent, env=env, verbose=True, feedback_window=window,
                )
                plt.close("all")
                save_history(episode_history, str(experiment_dir),
                             filename=f"{name}_s0_history.csv")
                save_feedback_log(feedback_log, str(experiment_dir),
                                  filename=f"{name}_s0_feedback_log.csv")
                agent.save(str(experiment_dir / f"{name}_s0_model.npz"))
                env.close()
                print("done")

        # Canonical (seed-0) copies for backward compat with visual_compare etc.
        import shutil
        s0_hist = experiment_dir / f"{name}_s0_history.csv"
        s0_model = experiment_dir / f"{name}_s0_model.npz"
        if s0_hist.exists():
            shutil.copy(s0_hist, experiment_dir / f"{name}_episode_history.csv")
        if s0_model.exists():
            shutil.copy(s0_model, experiment_dir / f"{name}_model.npz")

    print("\n" + "=" * 60)
    print(f"  All {len(conditions)} conditions × {len(SEEDS)} seeds completed!")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate_model(agent: QLearningAgent, num_episodes: int = 100) -> list[int]:
    """Run a trained agent with no step limit and return episode lengths."""
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


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

def _load_seed_histories(experiment_dir: pathlib.Path, base_dir: pathlib.Path,
                          max_episodes: int) -> dict:
    """
    Load per-seed episode histories for all conditions.
    Returns dict: key → list of DataFrames (one per seed).
    Falls back to the canonical (seed-0) file when per-seed files are absent
    so the function works with both old single-seed and new multi-seed outputs.
    """
    n = max_episodes
    ALL_META = [
        {"key": "baseline_shaped", "label": "Baseline",            "color": "blue",
         "is_hcrl": False, "dir": base_dir,       "prefix": "baseline"},
        {"key": "early",           "label": "HCRL Early (0-20%)",  "color": "green",
         "is_hcrl": True,  "dir": experiment_dir, "prefix": "early",
         "window": (0, int(0.2 * n))},
        {"key": "mid",             "label": "HCRL Mid (40-60%)",   "color": "orange",
         "is_hcrl": True,  "dir": experiment_dir, "prefix": "mid",
         "window": (int(0.4 * n), int(0.6 * n))},
        {"key": "late",            "label": "HCRL Late (80-100%)", "color": "purple",
         "is_hcrl": True,  "dir": experiment_dir, "prefix": "late",
         "window": (int(0.8 * n), n)},
        {"key": "full_feedback",   "label": "HCRL Full Feedback",  "color": "red",
         "is_hcrl": True,  "dir": experiment_dir, "prefix": "full_feedback",
         "window": (0, n)},
    ]

    result = {}
    for m in ALL_META:
        dfs = []
        for seed in SEEDS:
            p = m["dir"] / f"{m['prefix']}_s{seed}_history.csv"
            if p.exists():
                dfs.append(pd.read_csv(p, index_col="episode_index"))
        # fallback: old-style single file
        if not dfs:
            fallbacks = [
                m["dir"] / f"{m['prefix']}_episode_history.csv",  # timing cond
                m["dir"] / "episode_history.csv",                  # baseline
            ]
            for fb in fallbacks:
                if fb.exists():
                    dfs.append(pd.read_csv(fb, index_col="episode_index"))
                    break
        if dfs:
            result[m["key"]] = {"meta": m, "dfs": dfs}

    return result


def analyze_experiment(max_episodes: int, experiment_dir: pathlib.Path, eval_episodes: int = 100):
    """Load results (multi-seed aware) and generate comparison charts + significance tests."""
    from scipy import stats

    base_dir = experiment_dir.parent
    seed_data = _load_seed_histories(experiment_dir, base_dir, max_episodes)

    if len(seed_data) < 2:
        print("Need at least 2 conditions. Train first.")
        return

    available = list(seed_data.values())

    # --- Aggregate training stats across seeds ---
    def agg(dfs):
        """Return (overall_mean, overall_std, last30_mean, last30_std) across all seeds."""
        all_ep = pd.concat(dfs)["episode_length"]
        last30 = pd.concat([df["episode_length"].tail(30) for df in dfs])
        return all_ep.mean(), all_ep.std(), last30.mean(), last30.std()

    # --- Summary table ---
    col_width = 20
    n_cols = len(available)
    table_width = 24 + col_width * n_cols
    seeds_str = f"{len(SEEDS)} seed{'s' if len(SEEDS) > 1 else ''}"
    print("\n" + "=" * table_width)
    print(f"  {'TIMING EXPERIMENT — {} eps, {}'.format(max_episodes, seeds_str):^{table_width-2}}")
    print("=" * table_width)
    header = f"  {'Metric':<24}" + "".join(f"{d['meta']['label']:>{col_width}}" for d in available)
    print(header)
    print("  " + "-" * (table_width - 2))

    for stat_name, fn in [
        ("Train Mean ± Std",   lambda d: f"{agg(d['dfs'])[0]:.1f}±{agg(d['dfs'])[1]:.1f}"),
        ("Last-30 Avg ± Std",  lambda d: f"{agg(d['dfs'])[2]:.1f}±{agg(d['dfs'])[3]:.1f}"),
        ("N seeds",            lambda d: f"{len(d['dfs'])}"),
    ]:
        row = f"  {stat_name:<24}" + "".join(f"{fn(d):>{col_width}}" for d in available)
        print(row)
    print("=" * table_width)

    # --- Statistical significance: Mann-Whitney U vs baseline ---
    baseline_key = "baseline_shaped"
    if baseline_key in seed_data:
        base_lengths = pd.concat(seed_data[baseline_key]["dfs"])["episode_length"].values
        print(f"\n  Statistical tests (Mann-Whitney U vs Baseline, {seeds_str}):")
        print(f"  {'Condition':<26} {'Mean':>8} {'U-stat':>10} {'p-value':>10}  {'sig?':>6}")
        print("  " + "-" * 64)
        for d in available:
            key = d["meta"]["key"]
            if key == baseline_key:
                continue
            cond_lengths = pd.concat(d["dfs"])["episode_length"].values
            u_stat, p_val = stats.mannwhitneyu(cond_lengths, base_lengths, alternative="greater")
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            print(f"  {d['meta']['label']:<26} {np.mean(cond_lengths):>8.1f} {u_stat:>10.0f} {p_val:>10.4f}  {sig:>6}")
        print("  (*** p<0.001  ** p<0.01  * p<0.05  ns = not significant)")

    # --- Evaluate saved models ---
    print(f"\nEvaluating models ({eval_episodes} eps each, best-seed model)...")
    gameplay_results: dict[str, list[int]] = {}
    for d in available:
        m = d["meta"]
        # Try seed-0 model first, then canonical
        candidates = [
            m["dir"] / f"{m['prefix']}_s0_model.npz",
            m["dir"] / f"{m['prefix']}_model.npz",
            m["dir"] / "baseline_model.npz",
        ]
        for path in candidates:
            if path.exists():
                agent = QLearningAgent.load(path)
                lengths = evaluate_model(agent, eval_episodes)
                gameplay_results[m["key"]] = lengths
                print(f"  {m['label']}: mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}")
                break

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Timing Experiment: Baseline vs HCRL Conditions ({max_episodes} eps, {seeds_str})",
        fontsize=13, fontweight="bold",
    )

    # 1. Learning curves (mean ± std across seeds)
    ax = axes[0, 0]
    roll = 10
    for d in available:
        m = d["meta"]
        dfs = d["dfs"]
        min_len = min(len(df) for df in dfs)
        stacked = np.stack([df["episode_length"].values[:min_len] for df in dfs])
        mean_curve = pd.Series(stacked.mean(axis=0)).rolling(window=roll, min_periods=1).mean()
        std_curve  = pd.Series(stacked.std(axis=0)).rolling(window=roll, min_periods=1).mean()
        x = np.arange(min_len)
        ls = "--" if not m["is_hcrl"] else "-"
        ax.plot(x, mean_curve, color=m["color"], linewidth=2, linestyle=ls,
                label=f"{m['label']} (n={len(dfs)})")
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        color=m["color"], alpha=0.12)
        if m["is_hcrl"] and "window" in m:
            ax.axvspan(m["window"][0], m["window"][1], color=m["color"], alpha=0.07)
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5, label="Goal: 195")
    ax.set_title(f"Training Curves (mean ± std, {seeds_str})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Gameplay box plot
    ax = axes[0, 1]
    if gameplay_results:
        data   = [gameplay_results[d["meta"]["key"]] for d in available if d["meta"]["key"] in gameplay_results]
        labels = [d["meta"]["label"]                 for d in available if d["meta"]["key"] in gameplay_results]
        cols   = [d["meta"]["color"]                 for d in available if d["meta"]["key"] in gameplay_results]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for box, c in zip(bp["boxes"], cols):
            box.set_facecolor(c); box.set_alpha(0.4)
        ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5, label="Goal: 195")
        ax.set_title("Gameplay Distribution (seed-0 model)")
        ax.set_ylabel("Episode Length")
        ax.tick_params(axis="x", labelrotation=15)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 3. Mean training performance bar chart (mean ± std across seeds)
    ax = axes[1, 0]
    labels_bar = [d["meta"]["label"] for d in available]
    means_bar  = [agg(d["dfs"])[2]  for d in available]   # last-30 mean
    stds_bar   = [agg(d["dfs"])[3]  for d in available]   # last-30 std
    cols_bar   = [d["meta"]["color"] for d in available]
    bars = ax.bar(labels_bar, means_bar, color=cols_bar, alpha=0.75,
                  yerr=stds_bar, capsize=5)
    for bar, mv in zip(bars, means_bar):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"{mv:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5, label="Goal: 195")
    ax.set_title(f"Last-30 Training Avg ± Std ({seeds_str})")
    ax.set_ylabel("Episode Length")
    ax.tick_params(axis="x", labelrotation=15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Feedback density
    ax = axes[1, 1]
    for d in available:
        m = d["meta"]
        if not m["is_hcrl"]:
            continue
        # aggregate feedback across seeds
        all_counts: list[float] = []
        for seed in SEEDS:
            fb_path = experiment_dir / f"{m['prefix']}_s{seed}_feedback_log.csv"
            if not fb_path.exists():
                fb_path = experiment_dir / f"{m['key']}_feedback_log.csv"
            if fb_path.exists():
                fb_df = pd.read_csv(fb_path)
                if len(fb_df) > 0:
                    fb_per_ep = fb_df.groupby("episode").size()
                    counts = [fb_per_ep.get(ep, 0) for ep in range(max_episodes)]
                    all_counts.append(counts)
        if all_counts:
            mean_counts = pd.Series(np.mean(all_counts, axis=0)).rolling(window=5, min_periods=1).mean()
            total = int(np.sum(all_counts))
            ax.plot(range(max_episodes), mean_counts, color=m["color"], linewidth=2,
                    label=f"{m['label']} ({total} total)")
    ax.set_title("Feedback Density per Episode (HCRL only)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Feedback count (MA-5)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = experiment_dir / "timing_experiment_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100,
                        help="Total training episodes per condition (default: 100)")
    parser.add_argument("--auto", action="store_true",
                        help="Use oracle (automated) feedback instead of human keyboard input")
    parser.add_argument("--analyze", action="store_true",
                        help="Skip training, only run analysis on existing results")
    parser.add_argument("--skip-charts", action="store_true",
                        help="Skip chart generation and plt.show()")
    args = parser.parse_args()

    experiment_dir = pathlib.Path(f"experiment-results/ep{args.episodes}/timing-experiment")

    if not args.analyze:
        run_experiment(max_episodes=args.episodes, auto=args.auto, experiment_dir=experiment_dir)

    if not args.skip_charts:
        analyze_experiment(max_episodes=args.episodes, experiment_dir=experiment_dir)
