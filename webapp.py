"""
CartPole HCRL Web Visualizer
=============================
Watch trained CartPole models play directly in your browser.

Usage:
    uv run python webapp.py
    Open: http://localhost:5000
"""

import base64
import io
import json
import pathlib
import re
import time

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, stream_with_context
from PIL import Image

from cartpole.agents import QLearningAgent

app = Flask(__name__)
app.config["PROPAGATE_EXCEPTIONS"] = True
RESULTS_DIR = pathlib.Path("experiment-results")


@app.after_request
def _ngrok_headers(response):
    # Skip ngrok's browser-warning interstitial page
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

_NAME_MAP = {
    "baseline":      "Baseline",
    "early":         "Early (0-20%)",
    "mid":           "Mid (40-60%)",
    "late":          "Late (80-100%)",
    "full_feedback": "Full Feedback",
    "hcrl":          "HCRL (interactive)",
}


def make_label(npz: pathlib.Path) -> str:
    """Human-readable label derived from a model file path."""
    try:
        rel = npz.relative_to(RESULTS_DIR)
    except ValueError:
        rel = npz

    ep = next((p for p in rel.parts if re.match(r"ep\d+$", p)), "")
    ep_tag = f" ({ep})" if ep else ""
    stem = re.sub(r"_model$", "", npz.stem)

    # w20_s1  →  Weight=20 s1 (ep200)
    m = re.match(r"w(\d+)_s(\d+)$", stem)
    if m:
        return f"Weight={m.group(1)} s{m.group(2)}{ep_tag}"

    # early_s0  →  Early (0-20%) s0 (ep200)
    m = re.match(r"(early|mid|late|full_feedback)_s(\d+)$", stem)
    if m:
        return f"{_NAME_MAP[m.group(1)]} s{m.group(2)}{ep_tag}"

    # baseline_s2  →  Baseline s2 (ep200)
    m = re.match(r"baseline_s(\d+)$", stem)
    if m:
        return f"Baseline s{m.group(1)}{ep_tag}"

    return _NAME_MAP.get(stem, stem.replace("_", " ").title()) + ep_tag


_REWARD_MODEL_PATTERNS = re.compile(
    r"(^|_)(reward_model|hcrl_reward_model)(\.npz)?$", re.IGNORECASE
)


def _is_agent_model(npz: pathlib.Path) -> bool:
    """Return True only if the .npz file is a QLearningAgent (has q_table key)."""
    if _REWARD_MODEL_PATTERNS.search(npz.stem):
        return False
    try:
        with np.load(npz) as data:
            return "q_table" in data
    except Exception:
        return False


def scan_models() -> list[dict]:
    """Recursively find all QLearningAgent .npz files and return structured metadata."""
    if not RESULTS_DIR.exists():
        return []
    models = []
    for npz in sorted(RESULTS_DIR.rglob("*.npz")):
        if not _is_agent_model(npz):
            continue
        rel = npz.relative_to(RESULTS_DIR)
        ep = next((p for p in rel.parts if re.match(r"ep\d+$", p)), "misc")
        category = npz.parent.name if npz.parent.name != RESULTS_DIR.name else "root"
        models.append({
            "path":     str(npz).replace("\\", "/"),
            "label":    make_label(npz),
            "ep":       ep,
            "category": category,
            "group":    f"{ep} / {category}",
        })
    return models


# ---------------------------------------------------------------------------
# Gameplay streaming
# ---------------------------------------------------------------------------

def _encode_frame(env: gym.Env, max_w: int = 480, quality: int = 82) -> str:
    """Render one frame → base64 JPEG string."""
    frame = env.render()
    h, w = frame.shape[:2]
    if w > max_w:
        new_w = max_w
        new_h = int(h * max_w / w)
        img = Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS)
    else:
        img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def stream_gameplay(model_paths: list[str], num_episodes: int, fps: int):
    """
    Generator that drives all selected models forward in lock-step and yields
    SSE events containing base64-encoded frames + live stats.
    """
    agents, envs, labels = [], [], []

    for path in model_paths:
        p = pathlib.Path(path)
        if not _is_agent_model(p):
            raise ValueError(
                f"{p.name} is not a QLearningAgent model (missing q_table). "
                "Reward model .npz files cannot be played in the web visualizer."
            )
        agents.append(QLearningAgent.load(p))
        envs.append(gym.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=500))
        labels.append(make_label(p))

    n = len(model_paths)
    history = [[] for _ in range(n)]
    frame_dt = 1.0 / fps
    # Scale frames down when many models are shown simultaneously
    max_w = 480 if n <= 2 else (360 if n <= 4 else 280)

    try:
        for ep in range(num_episodes):
            observations, actions = [], []
            for agent, env in zip(agents, envs):
                obs, _ = env.reset()
                observations.append(obs)
                actions.append(agent.begin_episode(obs))

            dones = [False] * n
            steps = [0] * n

            while not all(dones):
                t0 = time.perf_counter()

                for i in range(n):
                    if not dones[i]:
                        obs, _, term, trunc, _ = envs[i].step(actions[i])
                        observations[i] = obs
                        steps[i] += 1
                        if term or trunc:
                            dones[i] = True
                            history[i].append(steps[i])
                        else:
                            actions[i] = agents[i].act(obs, reward=0.0)

                frames = [_encode_frame(env, max_w=max_w) for env in envs]
                stats = [
                    {
                        "label":     labels[i],
                        "episode":   ep + 1,
                        "steps":     steps[i],
                        "done":      dones[i],
                        "mean":      round(float(np.mean(history[i])), 1) if history[i] else 0,
                        "best":      int(max(history[i])) if history[i] else 0,
                        "completed": len(history[i]),
                    }
                    for i in range(n)
                ]
                yield _sse({
                    "type":    "frame",
                    "episode": ep + 1,
                    "total":   num_episodes,
                    "frames":  frames,
                    "stats":   stats,
                })

                spare = frame_dt - (time.perf_counter() - t0)
                if spare > 0:
                    time.sleep(spare)

            time.sleep(0.4)  # brief pause between episodes

        # Final summary sent once after all episodes complete
        summary = [
            {
                "label":     labels[i],
                "mean":      round(float(np.mean(history[i])), 1) if history[i] else 0,
                "median":    round(float(np.median(history[i])), 1) if history[i] else 0,
                "best":      int(max(history[i])) if history[i] else 0,
                "worst":     int(min(history[i])) if history[i] else 0,
                "goal_rate": round(
                    sum(1 for x in history[i] if x >= 195) / len(history[i]) * 100, 1
                ) if history[i] else 0,
                "history":   history[i],
            }
            for i in range(n)
        ]
        yield _sse({"type": "done", "summary": summary})

    except GeneratorExit:
        pass
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return _HTML


@app.route("/logo")
def logo():
    """Serve the HUST logo from the project root if present."""
    import mimetypes
    from flask import send_file, abort
    for name in ("hust_logo.png", "hust_logo.jpg", "hust_logo.jpeg", "hust_logo.svg"):
        p = pathlib.Path(name)
        if p.exists():
            mime = mimetypes.guess_type(name)[0] or "image/png"
            return send_file(p, mimetype=mime)
    abort(404)


@app.route("/api/models")
def api_models():
    return jsonify(scan_models())


@app.route("/api/play")
def api_play():
    paths    = request.args.getlist("models")
    episodes = max(1, min(int(request.args.get("episodes", 5)), 50))
    fps      = max(5, min(int(request.args.get("fps", 30)), 60))

    if not paths:
        return jsonify({"error": "No models selected"}), 400

    @stream_with_context
    def generate():
        yield from stream_gameplay(paths, episodes, fps)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# CSV discovery for charts
# ---------------------------------------------------------------------------

_HISTORY_PATTERN = re.compile(r"_history\.csv$", re.IGNORECASE)


def _csv_label(csv_path: pathlib.Path) -> str:
    """Human-readable label from a history CSV path."""
    try:
        rel = csv_path.relative_to(RESULTS_DIR)
    except ValueError:
        rel = csv_path
    stem = csv_path.stem.replace("_history", "").replace("_episode", "")
    ep = next((p for p in rel.parts if re.match(r"ep\d+$", p)), "")
    ep_tag = f" ({ep})" if ep else ""
    nice = _NAME_MAP.get(stem, stem.replace("_", " ").title())
    parent = csv_path.parent.name
    if parent not in ("experiment-results", ep.replace("ep", "ep")) and parent != RESULTS_DIR.name:
        nice = f"{parent}/{nice}"
    return nice + ep_tag


def _csv_family(csv_path: pathlib.Path) -> str:
    """Return a seed-stripped family key for grouping multi-seed CSVs."""
    stem = csv_path.stem  # e.g. baseline_s0_history
    # Strip seed suffix: "_s0_history" → "_history"
    base = re.sub(r"_s\d+", "", stem)  # e.g. baseline_history
    parent = csv_path.parent
    try:
        rel_parent = parent.relative_to(RESULTS_DIR)
    except ValueError:
        rel_parent = parent
    return str(rel_parent / base).replace("\\", "/")


def scan_csvs() -> list[dict]:
    """Find all *_history.csv files and return metadata."""
    if not RESULTS_DIR.exists():
        return []
    csvs = []
    for p in sorted(RESULTS_DIR.rglob("*_history.csv")):
        try:
            df = pd.read_csv(p, nrows=2)
            if "episode_length" not in df.columns:
                continue
        except Exception:
            continue
        rel = p.relative_to(RESULTS_DIR)
        ep = next((part for part in rel.parts if re.match(r"ep\d+$", part)), "misc")
        category = p.parent.name if p.parent.name != RESULTS_DIR.name else "root"
        csvs.append({
            "path":     str(p).replace("\\", "/"),
            "label":    _csv_label(p),
            "ep":       ep,
            "category": category,
            "group":    f"{ep} / {category}",
            "family":   _csv_family(p),
        })
    return csvs


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

_CHART_COLORS = [
    "#4361ee", "#e63946", "#2dc653", "#f77f00", "#7209b7",
    "#3a86a7", "#fb5607", "#8338ec", "#06d6a0", "#ef476f",
    "#118ab2", "#ffd166", "#073b4c", "#ff006e", "#8ac926",
]


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _load_history(path_str: str) -> pd.DataFrame | None:
    p = pathlib.Path(path_str)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if "episode_length" not in df.columns:
            return None
        return df
    except Exception:
        return None


# Style registry: map family-key *patterns* to display properties.
# Patterns are matched against the seed-stripped filename stem
# (e.g. "baseline_history", "early_history", "rlhf_oracle_history").
_MODEL_STYLES: list[tuple[str, dict]] = [
    (r"baseline",           {"label": "Baseline",              "color": "blue",       "linestyle": "--"}),
    (r"early",              {"label": "HCRL Early (0-20%)",    "color": "green",      "linestyle": "-"}),
    (r"mid",                {"label": "HCRL Mid (40-60%)",     "color": "orange",     "linestyle": "-"}),
    (r"late",               {"label": "HCRL Late (80-100%)",   "color": "purple",     "linestyle": "-"}),
    (r"full_feedback",      {"label": "HCRL Full Feedback",    "color": "red",        "linestyle": "-"}),
    (r"hcrl_oracle",        {"label": "HCRL Oracle",           "color": "darkgreen",  "linestyle": "-"}),
    (r"hcrl_human",         {"label": "HCRL Human",            "color": "limegreen",  "linestyle": "-."}),
    (r"rlhf_oracle",        {"label": "RLHF Oracle",           "color": "darkorange", "linestyle": "-"}),
    (r"rlhf_ensemble",      {"label": "RLHF Ensemble",         "color": "brown",      "linestyle": "-"}),
    (r"rlhf_human",         {"label": "RLHF Human",            "color": "salmon",     "linestyle": "-."}),
    (r"vi_tamer_human",     {"label": "VI-TAMER Human",        "color": "teal",       "linestyle": "-."}),
    (r"vi_tamer",           {"label": "VI-TAMER",              "color": "darkcyan",   "linestyle": "-"}),
]


def _model_style(family_key: str, fallback_idx: int) -> dict:
    """Return {label, color, linestyle} for a family key."""
    # family_key looks like "ep100/timing-experiment/early_history"
    # Extract the stem part after the last '/'
    stem = family_key.rsplit("/", 1)[-1]  # e.g. "early_history"
    stem = stem.replace("_history", "").replace("_episode", "")  # e.g. "early"
    for pattern, style in _MODEL_STYLES:
        if re.search(pattern, stem):
            return style
    # Fallback
    return {
        "label": stem.replace("_", " ").title(),
        "color": _CHART_COLORS[fallback_idx % len(_CHART_COLORS)],
        "linestyle": "-",
    }


def _group_seeds(paths: list[str]) -> list[tuple[str, dict, list[pd.DataFrame]]]:
    """Group CSVs that share the same model family (differ only by seed suffix).

    Returns [(family_key, style_dict, [df, ...])] preserving order of first occurrence.
    """
    import collections
    groups: dict[str, list[pd.DataFrame]] = collections.OrderedDict()
    for p in paths:
        df = _load_history(p)
        if df is None:
            continue
        pp = pathlib.Path(p)
        family = _csv_family(pp)
        groups.setdefault(family, []).append(df)
    result = []
    for i, (fam, dfs) in enumerate(groups.items()):
        style = _model_style(fam, i)
        result.append((fam, style, dfs))
    return result


def generate_chart(chart_type: str, csv_paths: list[str], options: dict) -> str | None:
    """Generate a chart and return base64-encoded PNG string."""
    window = int(options.get("window", 10))

    if chart_type == "training_curves":
        return _chart_training_curves(csv_paths, window)
    elif chart_type == "training_curves_std":
        return _chart_training_curves_std(csv_paths, window)
    elif chart_type == "box_plot":
        return _chart_box_plot(csv_paths)
    elif chart_type == "bar_chart":
        return _chart_bar_chart(csv_paths)
    elif chart_type == "histogram":
        return _chart_histogram(csv_paths)
    elif chart_type == "convergence":
        return _chart_convergence(csv_paths, window)
    elif chart_type == "success_rate":
        return _chart_success_rate(csv_paths, window)
    elif chart_type == "improvement_speed":
        return _chart_improvement_speed(csv_paths, window)
    elif chart_type == "stability":
        return _chart_stability(csv_paths, window)
    elif chart_type == "final_performance":
        return _chart_final_performance(csv_paths, window)
    elif chart_type == "heatmap":
        return _chart_heatmap(csv_paths)
    return None


def _chart_training_curves(paths: list[str], window: int) -> str:
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, p in enumerate(paths):
        df = _load_history(p)
        if df is None:
            continue
        color = _CHART_COLORS[i % len(_CHART_COLORS)]
        label = _csv_label(pathlib.Path(p))
        lengths = df["episode_length"].values
        rolled = pd.Series(lengths).rolling(window=window, min_periods=1).mean()
        ax.plot(rolled.index, rolled, label=label, color=color, linewidth=2)
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5, label="Goal: 195")
    ax.set_title(f"Training Curves (rolling mean, window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _chart_training_curves_std(paths: list[str], window: int) -> str:
    grouped = _group_seeds(paths)
    if not grouped:
        return _chart_training_curves(paths, window)

    fig, ax = plt.subplots(figsize=(14, 6))
    total_seeds = 0
    for _fam, style, dfs in grouped:
        n_seeds = len(dfs)
        total_seeds = max(total_seeds, n_seeds)
        min_len = min(len(df) for df in dfs)
        stacked = np.stack([df["episode_length"].values[:min_len] for df in dfs])
        mean_c = pd.Series(stacked.mean(axis=0)).rolling(window=window, min_periods=1).mean()
        std_c = pd.Series(stacked.std(axis=0)).rolling(window=window, min_periods=1).mean()
        x = np.arange(min_len)
        ax.plot(x, mean_c, label=f"{style['label']} (n={n_seeds})",
                color=style["color"], linestyle=style["linestyle"], linewidth=2)
        ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                        color=style["color"], alpha=0.12)
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5, label="Goal: 195")
    seeds_str = f"{total_seeds} seeds" if total_seeds > 1 else "1 seed"
    ax.set_title(f"Training Curves (mean ± std, {seeds_str})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _chart_box_plot(paths: list[str]) -> str:
    data, labels, colors = [], [], []
    for i, p in enumerate(paths):
        df = _load_history(p)
        if df is None:
            continue
        data.append(df["episode_length"].values)
        labels.append(_csv_label(pathlib.Path(p)))
        colors.append(_CHART_COLORS[i % len(_CHART_COLORS)])

    fig, ax = plt.subplots(figsize=(max(10, len(data) * 2), 6))
    bp = ax.boxplot(data, tick_labels=[""] * len(data), patch_artist=True)
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_alpha(0.55)
    patches = [mpatches.Patch(facecolor=c, alpha=0.7, label=l) for c, l in zip(colors, labels)]
    goal_line = plt.Line2D([0], [0], color="gray", linestyle="--", alpha=0.7, label="Goal: 195")
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Episode Length Distribution — Box Plot")
    ax.set_ylabel("Episode Length")
    ax.legend(handles=patches + [goal_line], fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _chart_bar_chart(paths: list[str]) -> str:
    means, stds, labels, colors = [], [], [], []
    for i, p in enumerate(paths):
        df = _load_history(p)
        if df is None:
            continue
        lengths = df["episode_length"].values
        means.append(np.mean(lengths))
        stds.append(np.std(lengths))
        labels.append(_csv_label(pathlib.Path(p)))
        colors.append(_CHART_COLORS[i % len(_CHART_COLORS)])

    fig, ax = plt.subplots(figsize=(max(10, len(means) * 2), 6))
    bars = ax.bar(range(len(means)), means, color=colors, alpha=0.75, yerr=stds, capsize=5)
    for bar, mv in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{mv:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5)
    patches = [mpatches.Patch(facecolor=c, alpha=0.7, label=l) for c, l in zip(colors, labels)]
    goal_line = plt.Line2D([0], [0], color="gray", linestyle="--", alpha=0.7, label="Goal: 195")
    ax.set_xticks([])
    ax.set_title("Mean Episode Length ± Std")
    ax.set_ylabel("Episode Length")
    ax.legend(handles=patches + [goal_line], fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    return _fig_to_base64(fig)


def _chart_histogram(paths: list[str]) -> str:
    fig, ax = plt.subplots(figsize=(14, 6))
    patches_legend = []
    for i, p in enumerate(paths):
        df = _load_history(p)
        if df is None:
            continue
        color = _CHART_COLORS[i % len(_CHART_COLORS)]
        label = _csv_label(pathlib.Path(p))
        ax.hist(df["episode_length"].values, bins=25, alpha=0.4, color=color, label=label)
        patches_legend.append(mpatches.Patch(facecolor=color, alpha=0.5, label=label))
    goal_line = plt.Line2D([0], [0], color="gray", linestyle="--", alpha=0.7, label="Goal: 195")
    ax.axvline(x=195, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Episode Length Distribution — Histogram")
    ax.set_xlabel("Episode Length")
    ax.set_ylabel("Count")
    ax.legend(handles=patches_legend + [goal_line], fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _chart_convergence(paths: list[str], window: int) -> str:
    thresholds = [50, 100, 150, 195]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"Convergence Analysis — First Episode Crossing Thresholds (rolling window={window})",
        fontsize=13, fontweight="bold",
    )

    crossing_data = {}
    loaded = []
    for i, p in enumerate(paths):
        df = _load_history(p)
        if df is None:
            continue
        color = _CHART_COLORS[i % len(_CHART_COLORS)]
        label = _csv_label(pathlib.Path(p))
        loaded.append((label, color, df))
        crossings = {}
        rolling = df["episode_length"].rolling(window=window, min_periods=1).mean()
        for th in thresholds:
            crossed = rolling[rolling >= th]
            crossings[th] = int(crossed.index[0]) if not crossed.empty else None
        crossing_data[label] = crossings

    if not loaded:
        plt.close(fig)
        return ""

    # Left: learning curves with threshold markers
    ax = axes[0]
    for label, color, df in loaded:
        rolling = df["episode_length"].rolling(window=window, min_periods=1).mean()
        ax.plot(rolling.index, rolling, color=color, linewidth=2, label=label)
        for th in thresholds:
            ep = crossing_data[label][th]
            if ep is not None:
                ax.plot(ep, th, marker="x", color=color, markersize=8, markeredgewidth=2)
    for th in thresholds:
        c = "red" if th == 195 else "gray"
        ls = "--" if th == 195 else ":"
        ax.axhline(y=th, color=c, linestyle=ls, alpha=0.55, linewidth=1)
        ax.text(ax.get_xlim()[1] * 0.98, th + 1, str(th), color=c, fontsize=8, ha="right")
    ax.set_title(f"Learning Curves + Threshold Crossings (× marks)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Right: grouped bar chart
    ax = axes[1]
    n_models = len(loaded)
    n_th = len(thresholds)
    bar_w = 0.8 / max(n_models, 1)
    x = np.arange(n_th)
    max_ep = 1
    for i, (label, color, df) in enumerate(loaded):
        vals = []
        for th in thresholds:
            ep = crossing_data[label][th]
            val = ep if ep is not None else len(df) + 10
            vals.append(val)
            if ep is not None:
                max_ep = max(max_ep, ep)
        offset = (i - n_models / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=label, color=color, alpha=0.75)
        for bar, val, th in zip(bars, vals, thresholds):
            ep = crossing_data[label][th]
            txt = str(ep) if ep is not None else "N/A"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    txt, ha="center", va="bottom", fontsize=7, fontweight="bold",
                    color=color, rotation=45)
    ax.set_title("Episode of First Threshold Crossing (lower = faster)")
    ax.set_xlabel("Performance Threshold")
    ax.set_ylabel("Episode Number")
    ax.set_xticks(x)
    ax.set_xticklabels([f"≥ {t}" for t in thresholds])
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max_ep * 1.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return _fig_to_base64(fig)


def _chart_success_rate(paths: list[str], window: int) -> str:
    """Rolling success rate: % of episodes reaching >=195 steps over a window."""
    grouped = _group_seeds(paths)
    if not grouped:
        return ""

    fig, ax = plt.subplots(figsize=(14, 6))
    total_seeds = 0
    for _fam, style, dfs in grouped:
        n_seeds = len(dfs)
        total_seeds = max(total_seeds, n_seeds)
        min_len = min(len(df) for df in dfs)
        # For each seed, compute binary success then average across seeds
        success_arrays = []
        for df in dfs:
            lengths = df["episode_length"].values[:min_len]
            success = (lengths >= 195).astype(float)
            rolled = pd.Series(success).rolling(window=window, min_periods=1).mean() * 100
            success_arrays.append(rolled.values)
        stacked = np.stack(success_arrays)
        mean_c = stacked.mean(axis=0)
        std_c = stacked.std(axis=0)
        x = np.arange(min_len)
        ax.plot(x, mean_c, label=f"{style['label']} (n={n_seeds})",
                color=style["color"], linestyle=style["linestyle"], linewidth=2)
        if n_seeds > 1:
            ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                            color=style["color"], alpha=0.12)
    ax.axhline(y=100, color="gray", linestyle=":", alpha=0.3)
    ax.axhline(y=50, color="gray", linestyle=":", alpha=0.3)
    seeds_str = f"{total_seeds} seeds" if total_seeds > 1 else "1 seed"
    ax.set_title(f"Success Rate Over Time (rolling {window}-ep window, {seeds_str})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _chart_improvement_speed(paths: list[str], window: int) -> str:
    """Episode-over-episode improvement rate (derivative of rolling mean)."""
    grouped = _group_seeds(paths)
    if not grouped:
        return ""

    fig, ax = plt.subplots(figsize=(14, 6))
    total_seeds = 0
    for _fam, style, dfs in grouped:
        n_seeds = len(dfs)
        total_seeds = max(total_seeds, n_seeds)
        min_len = min(len(df) for df in dfs)
        diff_arrays = []
        for df in dfs:
            lengths = df["episode_length"].values[:min_len]
            rolled = pd.Series(lengths).rolling(window=window, min_periods=1).mean()
            diff = rolled.diff().fillna(0).values
            diff_arrays.append(diff)
        stacked = np.stack(diff_arrays)
        mean_c = stacked.mean(axis=0)
        std_c = stacked.std(axis=0)
        # Smooth the derivative for readability
        mean_smooth = pd.Series(mean_c).rolling(window=window, min_periods=1).mean()
        std_smooth = pd.Series(std_c).rolling(window=window, min_periods=1).mean()
        x = np.arange(min_len)
        ax.plot(x, mean_smooth, label=f"{style['label']} (n={n_seeds})",
                color=style["color"], linestyle=style["linestyle"], linewidth=2)
        if n_seeds > 1:
            ax.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth,
                            color=style["color"], alpha=0.12)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.4, linewidth=1)
    seeds_str = f"{total_seeds} seeds" if total_seeds > 1 else "1 seed"
    ax.set_title(f"Learning Speed — Improvement Rate (Δ rolling mean, window={window}, {seeds_str})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Improvement (steps/episode)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _chart_stability(paths: list[str], window: int) -> str:
    """Rolling standard deviation over time — lower = more stable."""
    grouped = _group_seeds(paths)
    if not grouped:
        return ""

    fig, ax = plt.subplots(figsize=(14, 6))
    total_seeds = 0
    for _fam, style, dfs in grouped:
        n_seeds = len(dfs)
        total_seeds = max(total_seeds, n_seeds)
        min_len = min(len(df) for df in dfs)
        std_arrays = []
        for df in dfs:
            lengths = df["episode_length"].values[:min_len]
            rolled_std = pd.Series(lengths).rolling(window=window, min_periods=2).std().fillna(0).values
            std_arrays.append(rolled_std)
        stacked = np.stack(std_arrays)
        mean_c = stacked.mean(axis=0)
        x = np.arange(min_len)
        ax.plot(x, mean_c, label=f"{style['label']} (n={n_seeds})",
                color=style["color"], linestyle=style["linestyle"], linewidth=2)
    seeds_str = f"{total_seeds} seeds" if total_seeds > 1 else "1 seed"
    ax.set_title(f"Training Stability — Rolling Std Dev (window={window}, {seeds_str})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length Std Dev (lower = more stable)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _chart_final_performance(paths: list[str], window: int) -> str:
    """Grouped bar chart: mean of last N episodes for each model family."""
    grouped = _group_seeds(paths)
    if not grouped:
        return ""

    last_n = max(window, 5)
    labels, means, stds, colors = [], [], [], []
    for _fam, style, dfs in grouped:
        # For each seed, take last N episodes, then average across seeds
        seed_means = []
        for df in dfs:
            lengths = df["episode_length"].values
            tail = lengths[-last_n:] if len(lengths) >= last_n else lengths
            seed_means.append(np.mean(tail))
        labels.append(f"{style['label']} (n={len(dfs)})")
        means.append(np.mean(seed_means))
        stds.append(np.std(seed_means) if len(seed_means) > 1 else 0)
        colors.append(style["color"])

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.8), 6))
    bars = ax.bar(range(len(labels)), means, color=colors, alpha=0.75,
                  yerr=stds, capsize=5)
    for bar, mv in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{mv:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5, label="Goal: 195")
    patches = [mpatches.Patch(facecolor=c, alpha=0.7, label=l) for c, l in zip(colors, labels)]
    goal_line = plt.Line2D([0], [0], color="gray", linestyle="--", alpha=0.7, label="Goal: 195")
    ax.set_xticks([])
    ax.set_title(f"Final Performance — Mean of Last {last_n} Episodes (across seeds)")
    ax.set_ylabel("Episode Length")
    ax.legend(handles=patches + [goal_line], fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    return _fig_to_base64(fig)


def _chart_heatmap(paths: list[str]) -> str:
    """Performance heatmap: model families vs key metrics."""
    grouped = _group_seeds(paths)
    if not grouped:
        return ""

    metric_names = ["Mean", "Median", "Max", "Std", "Success\nRate %", "Best\nWindow"]
    labels = []
    data_rows = []
    colors_list = []
    for _fam, style, dfs in grouped:
        # Combine all episodes across seeds for overall metrics
        all_lengths = np.concatenate([df["episode_length"].values for df in dfs])
        # Best rolling-10 window mean
        if len(all_lengths) >= 10:
            best_win = pd.Series(all_lengths).rolling(10, min_periods=1).mean().max()
        else:
            best_win = np.mean(all_lengths)
        success_rate = (all_lengths >= 195).sum() / len(all_lengths) * 100

        labels.append(style["label"])
        colors_list.append(style["color"])
        data_rows.append([
            np.mean(all_lengths),
            np.median(all_lengths),
            np.max(all_lengths),
            np.std(all_lengths),
            success_rate,
            best_win,
        ])

    data = np.array(data_rows)
    n_models = len(labels)
    n_metrics = len(metric_names)

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 1.5), max(4, n_models * 0.7 + 2)))

    # Normalize each column for color mapping (0-1)
    data_norm = np.zeros_like(data)
    for j in range(n_metrics):
        col = data[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            data_norm[:, j] = (col - cmin) / (cmax - cmin)
        else:
            data_norm[:, j] = 0.5
    # For Std column (index 3), invert: lower is better
    data_norm[:, 3] = 1.0 - data_norm[:, 3]

    im = ax.imshow(data_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Annotate cells with actual values
    for i in range(n_models):
        for j in range(n_metrics):
            val = data[i, j]
            fmt = f"{val:.1f}" if val < 1000 else f"{val:.0f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=9, fontweight="bold",
                    color="black" if 0.3 < data_norm[i, j] < 0.7 else "white")

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Performance Heatmap (green = better)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Relative (higher = better)")

    plt.tight_layout()
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Gameplay chart generation (from live gameplay data, not CSV)
# ---------------------------------------------------------------------------

_GAMEPLAY_CHART_COLORS = [
    "steelblue", "crimson", "green", "darkorange", "purple",
    "teal", "brown", "deeppink", "olive", "slategray",
]


def _gameplay_box_plot(models: list[dict]) -> str:
    n = len(models)
    fig, ax = plt.subplots(figsize=(max(8, n * 2), 6))
    data = [m["history"] for m in models]
    labels = [m["label"] for m in models]
    colors = [_GAMEPLAY_CHART_COLORS[i % len(_GAMEPLAY_CHART_COLORS)] for i in range(n)]
    bp = ax.boxplot(data, tick_labels=[""] * n, patch_artist=True)
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_alpha(0.55)
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5)
    patches = [mpatches.Patch(facecolor=c, alpha=0.7, label=l) for c, l in zip(colors, labels)]
    goal_line = plt.Line2D([0], [0], color="gray", linestyle="--", alpha=0.7, label="Goal: 195")
    ax.set_title("Gameplay — Episode Length Distribution")
    ax.set_ylabel("Episode Length")
    ax.legend(handles=patches + [goal_line], fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _gameplay_bar_chart(models: list[dict]) -> str:
    n = len(models)
    labels = [m["label"] for m in models]
    colors = [_GAMEPLAY_CHART_COLORS[i % len(_GAMEPLAY_CHART_COLORS)] for i in range(n)]
    means = [np.mean(m["history"]) for m in models]
    stds = [np.std(m["history"]) for m in models]
    fig, ax = plt.subplots(figsize=(max(8, n * 2), 6))
    bars = ax.bar(range(n), means, color=colors, alpha=0.75, yerr=stds, capsize=5)
    for bar, mv in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{mv:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5)
    patches = [mpatches.Patch(facecolor=c, alpha=0.7, label=l) for c, l in zip(colors, labels)]
    goal_line = plt.Line2D([0], [0], color="gray", linestyle="--", alpha=0.7, label="Goal: 195")
    ax.set_xticks([])
    ax.set_title("Gameplay — Mean Episode Length ± Std")
    ax.set_ylabel("Episode Length")
    ax.legend(handles=patches + [goal_line], fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    return _fig_to_base64(fig)


def _gameplay_histogram(models: list[dict]) -> str:
    n = len(models)
    labels = [m["label"] for m in models]
    colors = [_GAMEPLAY_CHART_COLORS[i % len(_GAMEPLAY_CHART_COLORS)] for i in range(n)]
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, m in enumerate(models):
        ax.hist(m["history"], bins=max(10, len(m["history"]) // 3), alpha=0.4,
                color=colors[i], label=labels[i])
    ax.axvline(x=195, color="gray", linestyle="--", alpha=0.5)
    patches = [mpatches.Patch(facecolor=c, alpha=0.5, label=l) for c, l in zip(colors, labels)]
    goal_line = plt.Line2D([0], [0], color="gray", linestyle="--", alpha=0.7, label="Goal: 195")
    ax.set_title("Gameplay — Episode Length Histogram")
    ax.set_xlabel("Episode Length")
    ax.set_ylabel("Count")
    ax.legend(handles=patches + [goal_line], fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _gameplay_episode_progression(models: list[dict]) -> str:
    """Line chart: episode length over gameplay episodes — shows consistency."""
    n = len(models)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, m in enumerate(models):
        color = _GAMEPLAY_CHART_COLORS[i % len(_GAMEPLAY_CHART_COLORS)]
        h = m["history"]
        ax.plot(range(1, len(h) + 1), h, color=color, linewidth=1.5, alpha=0.7,
                marker="o", markersize=4, label=m["label"])
    ax.axhline(y=195, color="gray", linestyle="--", alpha=0.5, label="Goal: 195")
    ax.set_title("Gameplay — Episode Progression")
    ax.set_xlabel("Gameplay Episode")
    ax.set_ylabel("Episode Length")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _gameplay_summary_heatmap(models: list[dict]) -> str:
    """Heatmap comparing models across key gameplay metrics."""
    metric_names = ["Mean", "Median", "Best", "Worst", "Std", "Success\nRate %"]
    labels = [m["label"] for m in models]
    data_rows = []
    for m in models:
        h = np.array(m["history"], dtype=float)
        sr = (h >= 195).sum() / len(h) * 100 if len(h) > 0 else 0
        data_rows.append([np.mean(h), np.median(h), np.max(h), np.min(h), np.std(h), sr])
    data = np.array(data_rows)
    n_models, n_metrics = data.shape

    fig, ax = plt.subplots(figsize=(max(9, n_metrics * 1.4), max(3.5, n_models * 0.7 + 2)))
    data_norm = np.zeros_like(data)
    for j in range(n_metrics):
        col = data[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            data_norm[:, j] = (col - cmin) / (cmax - cmin)
        else:
            data_norm[:, j] = 0.5
    # Invert Worst (index 3) and Std (index 4): lower is better
    data_norm[:, 3] = 1.0 - data_norm[:, 3]
    data_norm[:, 4] = 1.0 - data_norm[:, 4]

    im = ax.imshow(data_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    for i in range(n_models):
        for j in range(n_metrics):
            val = data[i, j]
            fmt = f"{val:.1f}" if val < 1000 else f"{val:.0f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=9, fontweight="bold",
                    color="black" if 0.3 < data_norm[i, j] < 0.7 else "white")
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Gameplay Performance Heatmap (green = better)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Relative (higher = better)")
    plt.tight_layout()
    return _fig_to_base64(fig)


_GAMEPLAY_CHART_TYPES = {
    "gp_box_plot":       ("Box Plot",              _gameplay_box_plot),
    "gp_bar_chart":      ("Bar Chart (Mean ± Std)", _gameplay_bar_chart),
    "gp_histogram":      ("Histogram",              _gameplay_histogram),
    "gp_progression":    ("Episode Progression",    _gameplay_episode_progression),
    "gp_heatmap":        ("Performance Heatmap",    _gameplay_summary_heatmap),
}


@app.route("/api/gameplay-chart", methods=["POST"])
def api_gameplay_chart():
    """Generate comparison charts from gameplay results data."""
    data = request.get_json(force=True)
    models = data.get("models", [])  # [{label, history: [int, ...]}, ...]
    chart_types = data.get("chart_types", list(_GAMEPLAY_CHART_TYPES.keys()))

    if not models or not any(m.get("history") for m in models):
        return jsonify({"error": "No gameplay data"}), 400

    results = []
    for ct in chart_types:
        if ct not in _GAMEPLAY_CHART_TYPES:
            results.append({"chart_type": ct, "error": "Unknown chart type"})
            continue
        nice_name, fn = _GAMEPLAY_CHART_TYPES[ct]
        try:
            img = fn(models)
            if img:
                results.append({"chart_type": ct, "title": nice_name, "image": img})
            else:
                results.append({"chart_type": ct, "title": nice_name, "error": "No data"})
        except Exception as exc:
            results.append({"chart_type": ct, "title": nice_name, "error": str(exc)})

    return jsonify({"charts": results})


# ---------------------------------------------------------------------------
# Chart API routes
# ---------------------------------------------------------------------------

@app.route("/api/csvs")
def api_csvs():
    return jsonify(scan_csvs())


@app.route("/api/chart", methods=["POST"])
def api_chart():
    data = request.get_json(force=True)
    chart_type = data.get("chart_type", "training_curves")
    csv_paths = data.get("csvs", [])
    options = data.get("options", {})

    if not csv_paths:
        return jsonify({"error": "No CSVs selected"}), 400

    try:
        img_b64 = generate_chart(chart_type, csv_paths, options)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    if not img_b64:
        return jsonify({"error": "Chart generation failed — no data"}), 400

    return jsonify({"image": img_b64, "chart_type": chart_type})


_ALL_CHART_TYPES = [
    "training_curves",
    "training_curves_std",
    "box_plot",
    "bar_chart",
    "histogram",
    "convergence",
    "success_rate",
    "improvement_speed",
    "stability",
    "final_performance",
    "heatmap",
]


@app.route("/api/multi-chart", methods=["POST"])
def api_multi_chart():
    """Generate several charts in one request and return a list of results."""
    data = request.get_json(force=True)
    csv_paths = data.get("csvs", [])
    chart_types = data.get("chart_types", _ALL_CHART_TYPES)
    options = data.get("options", {})

    if not csv_paths:
        return jsonify({"error": "No CSVs selected"}), 400

    results = []
    for ct in chart_types:
        try:
            img_b64 = generate_chart(ct, csv_paths, options)
            if img_b64:
                results.append({"chart_type": ct, "image": img_b64})
            else:
                results.append({"chart_type": ct, "error": "No data"})
        except Exception as exc:
            results.append({"chart_type": ct, "error": str(exc)})

    return jsonify({"charts": results})


# ---------------------------------------------------------------------------
# Single-file HTML / CSS / JS frontend
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CartPole HCRL Visualizer</title>
<style>
:root {
  --accent:      #4361ee;
  --accent-dark: #3650d0;
  --danger:      #e63946;
  --success:     #2dc653;
  --bg:          #f0f2f5;
  --card-bg:     #ffffff;
  --sidebar-bg:  #ffffff;
  --border:      #e0e4ea;
  --text:        #1a1d23;
  --muted:       #6b7280;
  --shadow:      0 1px 4px rgba(0,0,0,.08), 0 4px 16px rgba(0,0,0,.04);
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg); color: var(--text);
  display: flex; flex-direction: column; height: 100vh; overflow: hidden;
}

/* ── Header ── */
header {
  background: var(--accent); color: #fff;
  padding: 8px 20px; display: flex; align-items: center; gap: 14px;
  box-shadow: 0 2px 8px rgba(67,97,238,.35); flex-shrink: 0; z-index: 10;
}
.hdr-logo {
  height: 48px; width: 48px; object-fit: contain;
  background: #fff; border-radius: 8px; padding: 3px; flex-shrink: 0;
}
.hdr-logo-placeholder {
  width: 48px; height: 48px; background: rgba(255,255,255,.18);
  border-radius: 8px; display: flex; align-items: center; justify-content: center;
  font-size: 1.5rem; flex-shrink: 0;
}
.hdr-text { flex: 1; min-width: 0; }
header h1  { font-size: 1.0rem; font-weight: 700; white-space: nowrap; }
header p   { font-size: 0.72rem; opacity: .85; margin-top: 1px; }
.hdr-meta {
  text-align: right; flex-shrink: 0; font-size: .7rem; opacity: .85; line-height: 1.5;
}
.hdr-meta strong { display: block; font-size: .75rem; opacity: 1; }

/* ── Tab bar ── */
.tab-bar {
  display: flex; background: #fff; border-bottom: 2px solid var(--border);
  flex-shrink: 0; padding: 0 16px;
}
.tab-btn {
  padding: 10px 24px; font-size: .85rem; font-weight: 700; cursor: pointer;
  border: none; background: none; color: var(--muted);
  border-bottom: 3px solid transparent; margin-bottom: -2px;
  transition: all .15s; display: flex; align-items: center; gap: 6px;
}
.tab-btn:hover { color: var(--accent); }
.tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab-icon { font-size: 1rem; }

/* ── Two-column layout ── */
.layout { display: flex; flex: 1; overflow: hidden; }
.tab-page { display: none; flex: 1; overflow: hidden; }
.tab-page.active { display: flex; }

/* ── Sidebar ── */
.sidebar {
  width: 268px; background: var(--sidebar-bg);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column; flex-shrink: 0;
}
.sb-title {
  padding: 11px 16px 9px; font-size: .68rem; font-weight: 700;
  color: var(--muted); text-transform: uppercase; letter-spacing: .09em;
  border-bottom: 1px solid var(--border); flex-shrink: 0;
}
.model-list { flex: 1; overflow-y: auto; }
.grp-hdr {
  padding: 8px 16px 4px; font-size: .68rem; font-weight: 700;
  color: var(--accent); text-transform: uppercase; letter-spacing: .07em;
  background: #f7f8ff; border-top: 1px solid var(--border);
}
.model-item {
  display: flex; align-items: flex-start; padding: 7px 16px;
  cursor: pointer; transition: background .12s; gap: 8px;
}
.model-item:hover { background: #eef1ff; }
.model-item input  {
  accent-color: var(--accent); width: 14px; height: 14px;
  flex-shrink: 0; margin-top: 2px; cursor: pointer;
}
.model-item span { font-size: .81rem; line-height: 1.4; }
.no-models {
  padding: 28px 16px; text-align: center; color: var(--muted); font-size: .85rem;
}

/* ── Controls ── */
.controls {
  padding: 12px 14px; border-top: 1px solid var(--border);
  display: flex; flex-direction: column; gap: 11px; flex-shrink: 0;
}
.ctrl-row label {
  font-size: .72rem; font-weight: 700; color: var(--muted);
  display: flex; justify-content: space-between; margin-bottom: 4px;
}
.ctrl-row label strong { color: var(--accent); font-size: .8rem; }
input[type="range"] { width: 100%; accent-color: var(--accent); }
.btn {
  border: none; border-radius: 8px; padding: 9px;
  font-size: .88rem; font-weight: 700; cursor: pointer;
  display: flex; align-items: center; justify-content: center; gap: 6px;
  transition: all .15s; width: 100%; letter-spacing: .01em;
}
.btn-play { background: var(--accent); color: #fff; }
.btn-play:hover:not(:disabled) { background: var(--accent-dark); }
.btn-play:disabled { opacity: .4; cursor: not-allowed; }
.btn-stop { background: var(--danger); color: #fff; }
.btn-stop:hover { background: #c9313d; }

/* ── Main area ── */
.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.game-area { flex: 1; overflow-y: auto; padding: 14px; }

/* Empty hint */
.hint {
  height: 100%; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 10px; color: var(--muted); font-size: .9rem; text-align: center;
}
.hint-icon { font-size: 2.8rem; opacity: .3; }

/* ── Game grid ── */
.game-grid { display: grid; gap: 14px; }

/* ── Game card ── */
.game-card {
  background: var(--card-bg); border-radius: 12px;
  box-shadow: var(--shadow); border: 2px solid transparent;
  transition: border-color .25s; overflow: hidden;
}
.game-card.fell { border-color: var(--danger); }

.card-head {
  padding: 8px 12px; background: #fafbff;
  border-bottom: 1px solid var(--border);
  display: flex; justify-content: space-between; align-items: center; gap: 8px;
}
.card-name {
  font-size: .8rem; font-weight: 700; color: var(--accent);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.ep-badge {
  font-size: .68rem; background: #e8ecff; color: var(--accent);
  padding: 2px 8px; border-radius: 10px; font-weight: 700; flex-shrink: 0;
}

.frame-box {
  background: #0f172a;
  display: flex; align-items: center; justify-content: center; min-height: 160px;
}
.frame-box img { width: 100%; display: block; }
.frame-ph { color: #475569; font-size: .78rem; }

.card-stats {
  display: grid; grid-template-columns: repeat(3,1fr);
  border-top: 1px solid var(--border);
}
.stat {
  text-align: center; padding: 6px 0;
  border-right: 1px solid var(--border);
}
.stat:last-child { border-right: none; }
.stat-v { font-size: .98rem; font-weight: 700; }
.stat-k { font-size: .6rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }

/* ── Results ── */
.results {
  background: var(--card-bg); border-top: 2px solid var(--border);
  flex-shrink: 0; max-height: 180px; overflow-y: auto;
}
.res-head {
  padding: 7px 14px 5px; font-size: .68rem; font-weight: 700; color: var(--muted);
  text-transform: uppercase; letter-spacing: .09em;
  border-bottom: 1px solid var(--border); position: sticky; top: 0;
  background: var(--card-bg); z-index: 1;
}
table { width: 100%; border-collapse: collapse; font-size: .78rem; }
th {
  padding: 5px 12px; background: #f8f9fa; color: var(--muted);
  font-weight: 700; font-size: .68rem; text-transform: uppercase;
  letter-spacing: .04em; text-align: left; border-bottom: 1px solid var(--border);
}
td { padding: 5px 12px; border-bottom: 1px solid #f3f4f6; }
tr:last-child td { border-bottom: none; }
.tr-best { background: #f0fff4; }
.badge-best {
  background: #dcfce7; color: #15803d;
  padding: 1px 6px; border-radius: 8px; font-size: .67rem; font-weight: 700;
  margin-left: 4px;
}
.badge-goal {
  background: #dbeafe; color: #1d4ed8;
  padding: 1px 6px; border-radius: 8px; font-size: .67rem; font-weight: 700;
}
.res-empty { padding: 10px 14px; color: var(--muted); font-size: .8rem; }

/* ── Status bar ── */
.statusbar {
  padding: 5px 14px; background: #f8f9fa; border-top: 1px solid var(--border);
  font-size: .73rem; color: var(--muted);
  display: flex; justify-content: space-between; align-items: center;
  flex-shrink: 0;
}
.dot {
  width: 7px; height: 7px; border-radius: 50%; background: #d1d5db;
  display: inline-block; margin-right: 5px; vertical-align: middle;
}
.dot.live { background: var(--success); animation: blink 1.2s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }
.pbar-wrap { display: flex; align-items: center; gap: 8px; }
.pbar { width: 130px; height: 4px; background: #e5e7eb; border-radius: 2px; overflow: hidden; }
.pbar-fill { height: 100%; background: var(--accent); border-radius: 2px; transition: width .3s; }

/* ── Charts page ── */
.chart-sidebar {
  width: 300px; background: var(--sidebar-bg);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column; flex-shrink: 0;
}
.chart-main {
  flex: 1; display: flex; flex-direction: column; overflow: hidden;
}
.chart-display {
  flex: 1; overflow-y: auto; padding: 20px;
  display: flex; align-items: flex-start; justify-content: center;
}
.chart-display img {
  max-width: 100%; height: auto; border-radius: 10px;
  box-shadow: var(--shadow); background: #fff;
}
.chart-controls {
  padding: 12px 14px; border-top: 1px solid var(--border);
  display: flex; flex-direction: column; gap: 10px; flex-shrink: 0;
}
.chart-controls select {
  width: 100%; padding: 8px 10px; border-radius: 8px;
  border: 1px solid var(--border); font-size: .82rem;
  color: var(--text); background: #fff; cursor: pointer;
  appearance: auto;
}
.chart-controls select:focus { outline: 2px solid var(--accent); border-color: var(--accent); }
.chart-status {
  padding: 5px 14px; background: #f8f9fa; border-top: 1px solid var(--border);
  font-size: .73rem; color: var(--muted); flex-shrink: 0;
}
.csv-count {
  padding: 6px 14px; font-size: .72rem; color: var(--muted);
  border-top: 1px solid var(--border); flex-shrink: 0;
  display: flex; justify-content: space-between; align-items: center;
}
.csv-count a {
  color: var(--accent); cursor: pointer; text-decoration: none; font-weight: 600;
}
.csv-count a:hover { text-decoration: underline; }
.spinner {
  display: inline-block; width: 16px; height: 16px;
  border: 2px solid var(--border); border-top-color: var(--accent);
  border-radius: 50%; animation: spin .6s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.chart-desc {
  font-size: .72rem; color: var(--muted); line-height: 1.4;
  padding: 0 2px;
}

/* ── Multi-chart grid ── */
.chart-grid {
  display: grid; gap: 18px; width: 100%;
  grid-template-columns: 1fr;
}
.chart-grid.cols-2 { grid-template-columns: 1fr 1fr; }
.chart-card {
  background: var(--card-bg); border-radius: 12px;
  box-shadow: var(--shadow); overflow: hidden;
}
.chart-card-head {
  padding: 8px 14px; background: #fafbff;
  border-bottom: 1px solid var(--border);
  font-size: .78rem; font-weight: 700; color: var(--accent);
}
.chart-card img { width: 100%; display: block; }
.chart-card .chart-error {
  padding: 24px; text-align: center; color: var(--muted); font-size: .82rem;
}

/* ── Family group checkbox ── */
.grp-hdr-row {
  display: flex; align-items: center; gap: 6px;
  padding: 8px 16px 4px; font-size: .68rem; font-weight: 700;
  color: var(--accent); text-transform: uppercase; letter-spacing: .07em;
  background: #f7f8ff; border-top: 1px solid var(--border);
}
.grp-hdr-row input {
  accent-color: var(--accent); width: 13px; height: 13px; cursor: pointer;
}
.btn-row { display: flex; gap: 8px; }
.btn-row .btn { flex: 1; }
.btn-secondary {
  background: #e8ecff; color: var(--accent);
  border: none; border-radius: 8px; padding: 9px;
  font-size: .82rem; font-weight: 700; cursor: pointer;
  display: flex; align-items: center; justify-content: center; gap: 6px;
  transition: all .15s; letter-spacing: .01em;
}
.btn-secondary:hover:not(:disabled) { background: #d4dbff; }
.btn-secondary:disabled { opacity: .4; cursor: not-allowed; }

/* ── Chart type multi-select ── */
.chart-type-list {
  max-height: 180px; overflow-y: auto;
  border: 1px solid var(--border); border-radius: 8px;
  background: #fff; padding: 4px 0;
}
.ct-item {
  display: flex; align-items: center; gap: 7px;
  padding: 4px 12px; cursor: pointer; font-size: .78rem;
  transition: background .1s;
}
.ct-item:hover { background: #f0f2ff; }
.ct-item input { accent-color: var(--accent); width: 14px; height: 14px; cursor: pointer; }
.ct-item span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ct-count {
  font-size: .72rem; color: var(--muted); padding: 4px 2px 6px;
}

/* ── Gameplay charts section ── */
.gp-charts {
  margin-top: 2px; border-top: 1px solid var(--border);
  max-height: 600px; overflow-y: auto;
}
.gp-charts-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 14px; background: #f7f8ff;
  border-bottom: 1px solid var(--border);
}
.gp-charts-head span {
  font-size: .82rem; font-weight: 700; color: var(--accent);
}
.gp-charts-body { padding: 14px; }
</style>
</head>
<body>

<header>
  <img id="hdrLogo" class="hdr-logo" src="/logo" alt="HUST"
       onerror="this.style.display='none';document.getElementById('hdrIcon').style.display='flex'">
  <div id="hdrIcon" class="hdr-logo-placeholder" style="display:none">🎮</div>
  <div class="hdr-text">
    <h1>CartPole HCRL — Live Visualizer</h1>
    <p>Hanoi University of Science and Technology &nbsp;·&nbsp; Human-Centered Reinforcement Learning</p>
  </div>
  <div class="hdr-meta">
    <strong>Statistical Machine Learning</strong>
    Assoc. Prof. Thân Quang Khoát
  </div>
</header>

<!-- ── Tab bar ── -->
<nav class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('play')" id="tabPlay">
    <span class="tab-icon">🎮</span> Play
  </button>
  <button class="tab-btn" onclick="switchTab('charts')" id="tabCharts">
    <span class="tab-icon">📊</span> Charts
  </button>
</nav>

<!-- ══════════════════════════════════════════════════════════════ -->
<!-- TAB 1: PLAY (original gameplay visualizer)                    -->
<!-- ══════════════════════════════════════════════════════════════ -->
<div class="layout tab-page active" id="pagePlay">

  <!-- ── Sidebar ── -->
  <aside class="sidebar">
    <div class="sb-title">Select Models</div>
    <div class="model-list" id="modelList">
      <div class="no-models">Loading…</div>
    </div>
    <div class="controls">
      <div class="ctrl-row">
        <label>Episodes <strong id="epVal">5</strong></label>
        <input type="range" id="epSlider" min="1" max="30" value="5"
               oninput="epVal.textContent=this.value">
      </div>
      <div class="ctrl-row">
        <label>Speed <strong id="fpsVal">30</strong> fps</label>
        <input type="range" id="fpsSlider" min="5" max="60" value="30" step="5"
               oninput="fpsVal.textContent=this.value">
      </div>
      <button class="btn btn-play" id="playBtn" onclick="togglePlay()">▶ Play</button>
    </div>
  </aside>

  <!-- ── Main ── -->
  <main class="main">

    <div class="game-area" id="gameArea">
      <div class="hint">
        <div class="hint-icon">🤖</div>
        <div>Select one or more models from the sidebar,<br>then click <strong>Play</strong>.</div>
      </div>
    </div>

    <div class="results" id="resultsPanel">
      <div class="res-head">Results</div>
      <div class="res-empty" id="resBody">No results yet.</div>
    </div>

    <div class="gp-charts" id="gpChartsPanel" style="display:none">
      <div class="gp-charts-head">
        <span>Gameplay Comparison Charts</span>
        <button class="btn-secondary" id="gpChartBtn" onclick="generateGameplayCharts()" style="width:auto;padding:5px 14px;font-size:.75rem">
          📊 Generate Charts
        </button>
      </div>
      <div class="gp-charts-body" id="gpChartsBody">
        <div class="hint" style="padding:18px 0">
          <div style="color:var(--muted);font-size:.82rem">Click <strong>Generate Charts</strong> to compare model performance.</div>
        </div>
      </div>
    </div>

    <div class="statusbar">
      <div><span class="dot" id="dot"></span><span id="statusTxt">Ready</span></div>
      <div class="pbar-wrap">
        <div class="pbar"><div class="pbar-fill" id="pFill" style="width:0"></div></div>
        <span id="pTxt">—</span>
      </div>
    </div>

  </main>
</div>

<!-- ══════════════════════════════════════════════════════════════ -->
<!-- TAB 2: CHARTS (dynamic chart generation from CSV data)        -->
<!-- ══════════════════════════════════════════════════════════════ -->
<div class="layout tab-page" id="pageCharts">

  <!-- ── Chart sidebar ── -->
  <aside class="chart-sidebar">
    <div class="sb-title">Select CSV Data</div>
    <div class="model-list" id="csvList">
      <div class="no-models">Loading…</div>
    </div>
    <div class="csv-count">
      <span id="csvCountTxt">0 selected</span>
      <a onclick="toggleAllCsvs()">Select All</a>
    </div>
    <div class="chart-controls">
      <div class="ctrl-row" style="justify-content:space-between">
        <label>Chart Types</label>
        <a onclick="toggleAllChartTypes()" style="font-size:.72rem;cursor:pointer;color:var(--accent)">Select All</a>
      </div>
      <div class="chart-type-list" id="chartTypeList">
        <label class="ct-item"><input type="checkbox" value="training_curves" checked onchange="onChartTypeToggle()"><span>Training Curves</span></label>
        <label class="ct-item"><input type="checkbox" value="training_curves_std" checked onchange="onChartTypeToggle()"><span>Training Curves (Mean ± Std)</span></label>
        <label class="ct-item"><input type="checkbox" value="box_plot" onchange="onChartTypeToggle()"><span>Box Plot</span></label>
        <label class="ct-item"><input type="checkbox" value="bar_chart" onchange="onChartTypeToggle()"><span>Bar Chart (Mean ± Std)</span></label>
        <label class="ct-item"><input type="checkbox" value="histogram" onchange="onChartTypeToggle()"><span>Histogram</span></label>
        <label class="ct-item"><input type="checkbox" value="convergence" onchange="onChartTypeToggle()"><span>Convergence Analysis</span></label>
        <label class="ct-item"><input type="checkbox" value="success_rate" onchange="onChartTypeToggle()"><span>Success Rate Over Time</span></label>
        <label class="ct-item"><input type="checkbox" value="improvement_speed" onchange="onChartTypeToggle()"><span>Learning Speed</span></label>
        <label class="ct-item"><input type="checkbox" value="stability" onchange="onChartTypeToggle()"><span>Training Stability</span></label>
        <label class="ct-item"><input type="checkbox" value="final_performance" onchange="onChartTypeToggle()"><span>Final Performance</span></label>
        <label class="ct-item"><input type="checkbox" value="heatmap" onchange="onChartTypeToggle()"><span>Performance Heatmap</span></label>
      </div>
      <div class="ct-count" id="ctCountTxt">2 chart types selected</div>
      <div class="ctrl-row">
        <label>Rolling Window <strong id="winVal">10</strong></label>
        <input type="range" id="winSlider" min="1" max="50" value="10"
               oninput="winVal.textContent=this.value">
      </div>
      <button class="btn btn-play" id="chartBtn" onclick="generateSelectedCharts()">📊 Generate Charts</button>
    </div>
  </aside>

  <!-- ── Chart display ── -->
  <div class="chart-main">
    <div class="chart-display" id="chartDisplay">
      <div class="hint">
        <div class="hint-icon">📊</div>
        <div>Select CSV files from the sidebar, choose a chart type,<br>then click <strong>Generate Chart</strong>.</div>
      </div>
    </div>
    <div class="chart-status" id="chartStatus">Ready</div>
  </div>
</div>

<script>
/* ══════════════════════════════════════════════════════════════════
   TAB SWITCHING
   ══════════════════════════════════════════════════════════════════ */
function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-page').forEach(p => p.classList.remove('active'));
  if (tab === 'play') {
    document.getElementById('tabPlay').classList.add('active');
    document.getElementById('pagePlay').classList.add('active');
  } else {
    document.getElementById('tabCharts').classList.add('active');
    document.getElementById('pageCharts').classList.add('active');
    if (!chartsLoaded) loadCsvList();
  }
}

/* ══════════════════════════════════════════════════════════════════
   PLAY TAB (original code, unchanged)
   ══════════════════════════════════════════════════════════════════ */
let es = null;
let selected = new Set();
let labelMap  = {};
let lastGameplaySummary = null;

fetch('/api/models')
  .then(r => r.json())
  .then(models => {
    for (const m of models) labelMap[m.path] = m.label;
    renderSidebar(models);
  });

function renderSidebar(models) {
  const list = document.getElementById('modelList');
  if (!models.length) {
    list.innerHTML = '<div class="no-models">No models found.<br>Run experiments first.</div>';
    return;
  }
  const tree = {};
  for (const m of models) {
    if (!tree[m.group]) tree[m.group] = [];
    tree[m.group].push(m);
  }
  let html = '';
  for (const [group, items] of Object.entries(tree)) {
    html += `<div class="grp-hdr">${esc(group)}</div>`;
    for (const m of items) {
      html += `
        <label class="model-item">
          <input type="checkbox" value="${esc(m.path)}" onchange="onToggle(this)">
          <span>${esc(m.label)}</span>
        </label>`;
    }
  }
  list.innerHTML = html;
}

function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/"/g,'&quot;')
    .replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function onToggle(cb) {
  cb.checked ? selected.add(cb.value) : selected.delete(cb.value);
}

function togglePlay() { es ? stopPlay() : startPlay(); }

function startPlay() {
  if (!selected.size) { alert('Select at least one model first.'); return; }
  lastGameplaySummary = null;
  const gpPanel = document.getElementById('gpChartsPanel');
  if (gpPanel) gpPanel.style.display = 'none';
  const paths    = [...selected];
  const episodes = +document.getElementById('epSlider').value;
  const fps      = +document.getElementById('fpsSlider').value;
  const p = new URLSearchParams();
  paths.forEach(path => p.append('models', path));
  p.set('episodes', episodes);
  p.set('fps', fps);
  buildCards(paths);
  setProgress(0, episodes);
  setStatus('Connecting…');
  const btn = document.getElementById('playBtn');
  btn.className = 'btn btn-stop';
  btn.textContent = '■ Stop';
  document.getElementById('dot').classList.add('live');
  es = new EventSource('/api/play?' + p);
  es.onmessage = onMsg;
  es.onerror   = () => stopPlay(false);
}

function stopPlay(clearStatus = true) {
  if (es) { es.close(); es = null; }
  const btn = document.getElementById('playBtn');
  btn.className = 'btn btn-play';
  btn.textContent = '▶ Play';
  document.getElementById('dot').classList.remove('live');
  if (clearStatus) setStatus('Ready');
}

function onMsg(e) {
  const d = JSON.parse(e.data);
  if (d.type === 'frame') {
    d.frames.forEach((b64, i) => {
      const img = document.getElementById('f' + i);
      const ph  = document.getElementById('ph' + i);
      if (img) { img.src = 'data:image/jpeg;base64,' + b64; img.style.display = 'block'; }
      if (ph)  ph.style.display = 'none';
    });
    if (d.stats) d.stats.forEach((s, i) => updateCard(i, s));
    setProgress(d.episode, d.total);
    setStatus(`Episode ${d.episode} / ${d.total}`);
  } else if (d.type === 'done') {
    lastGameplaySummary = d.summary;
    showResults(d.summary);
    stopPlay(false);
    setStatus(`Done — ${d.summary.length} model(s) evaluated`);
    // Show gameplay charts panel and auto-generate
    const gpPanel = document.getElementById('gpChartsPanel');
    if (gpPanel && d.summary.length >= 1) {
      gpPanel.style.display = '';
      generateGameplayCharts();
    }
  }
}

function buildCards(paths) {
  const area = document.getElementById('gameArea');
  const n    = paths.length;
  const cols = n === 1 ? '1fr'
             : n === 2 ? '1fr 1fr'
             : 'repeat(auto-fill, minmax(280px, 1fr))';
  let html = `<div class="game-grid" style="grid-template-columns:${cols}">`;
  for (let i = 0; i < n; i++) {
    const lbl = esc(labelMap[paths[i]] || paths[i].split(/[/\\]/).pop());
    html += `
      <div class="game-card" id="card${i}">
        <div class="card-head">
          <span class="card-name" title="${lbl}">${lbl}</span>
          <span class="ep-badge" id="ep${i}">Ep —</span>
        </div>
        <div class="frame-box">
          <img id="f${i}" style="display:none" alt="game frame">
          <div id="ph${i}" class="frame-ph">Waiting for frames…</div>
        </div>
        <div class="card-stats">
          <div class="stat">
            <div class="stat-v" id="sv${i}">—</div>
            <div class="stat-k">Steps</div>
          </div>
          <div class="stat">
            <div class="stat-v" id="mv${i}">—</div>
            <div class="stat-k">Mean</div>
          </div>
          <div class="stat">
            <div class="stat-v" id="bv${i}">—</div>
            <div class="stat-k">Best</div>
          </div>
        </div>
      </div>`;
  }
  html += '</div>';
  area.innerHTML = html;
}

function updateCard(i, s) {
  const el = id => document.getElementById(id + i);
  const ep = el('ep');  if (ep)  ep.textContent  = `Ep ${s.episode}`;
  const sv = el('sv');  if (sv)  sv.textContent  = s.steps;
  const mv = el('mv');  if (mv)  mv.textContent  = s.mean > 0  ? s.mean.toFixed(1) : '—';
  const bv = el('bv');  if (bv)  bv.textContent  = s.best > 0  ? s.best : '—';
  const card = document.getElementById('card' + i);
  if (card) card.classList.toggle('fell', s.done);
}

function showResults(summary) {
  if (!summary?.length) return;
  const bestMean = Math.max(...summary.map(s => s.mean));
  let html = `<table>
    <thead><tr>
      <th>#</th><th>Model</th><th>Mean</th><th>Median</th>
      <th>Best</th><th>Worst</th><th>≥195 steps</th>
    </tr></thead><tbody>`;
  summary.forEach((s, i) => {
    const isBest = s.mean === bestMean;
    const goalBadge = s.goal_rate >= 50
      ? `<span class="badge-goal">${s.goal_rate}%</span>` : `${s.goal_rate}%`;
    html += `<tr${isBest ? ' class="tr-best"' : ''}>
      <td>${i + 1}</td>
      <td><strong>${esc(s.label)}</strong>${isBest ? '<span class="badge-best">Best</span>' : ''}</td>
      <td><strong>${s.mean}</strong></td>
      <td>${s.median}</td>
      <td>${s.best}</td>
      <td>${s.worst}</td>
      <td>${goalBadge}</td>
    </tr>`;
  });
  html += '</tbody></table>';
  document.getElementById('resBody').innerHTML = html;
}

function setStatus(t) { document.getElementById('statusTxt').textContent = t; }
function setProgress(cur, total) {
  document.getElementById('pFill').style.width = total ? (cur / total * 100) + '%' : '0';
  document.getElementById('pTxt').textContent  = total ? `${cur} / ${total} ep` : '—';
}

/* ── Gameplay Comparison Charts ── */
function generateGameplayCharts() {
  if (!lastGameplaySummary || !lastGameplaySummary.length) {
    alert('No gameplay data yet. Play some models first.');
    return;
  }
  const body = document.getElementById('gpChartsBody');
  const btn  = document.getElementById('gpChartBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Generating…';
  body.innerHTML = '<div class="hint" style="padding:18px 0"><div class="spinner" style="width:24px;height:24px;border-width:2px"></div><div style="color:var(--muted);font-size:.82rem">Generating comparison charts…</div></div>';

  const models = lastGameplaySummary.map(s => ({
    label: s.label,
    history: s.history,
  }));

  fetch('/api/gameplay-chart', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ models }),
  })
  .then(r => r.json())
  .then(data => {
    btn.disabled = false;
    btn.innerHTML = '📊 Generate Charts';
    if (data.error) {
      body.innerHTML = `<div class="hint" style="padding:18px 0"><div style="color:var(--muted)">⚠️ ${esc(data.error)}</div></div>`;
      return;
    }
    const charts = data.charts || [];
    const ok = charts.filter(c => c.image);
    const useCols2 = ok.length >= 2;
    let html = `<div class="chart-grid${useCols2 ? ' cols-2' : ''}">`;
    for (const c of charts) {
      const title = c.title || c.chart_type;
      html += `<div class="chart-card">`;
      html += `<div class="chart-card-head">${esc(title)}</div>`;
      if (c.image) {
        html += `<img src="data:image/png;base64,${c.image}" alt="${esc(title)}">`;
      } else {
        html += `<div class="chart-error">⚠️ ${esc(c.error || 'No data')}</div>`;
      }
      html += `</div>`;
    }
    html += '</div>';
    body.innerHTML = html;
  })
  .catch(err => {
    btn.disabled = false;
    btn.innerHTML = '📊 Generate Charts';
    body.innerHTML = `<div class="hint" style="padding:18px 0"><div style="color:var(--muted)">⚠️ ${esc(err.message)}</div></div>`;
  });
}

/* ══════════════════════════════════════════════════════════════════
   CHARTS TAB
   ══════════════════════════════════════════════════════════════════ */
let chartsLoaded = false;
let csvSelected  = new Set();
let allCsvPaths  = [];
let allCsvData   = [];      // full csv metadata from API
let familyMap    = {};       // family -> [csv objects]

const CHART_TYPES_NICE = {
  training_curves:     'Training Curves',
  training_curves_std: 'Training Curves (Mean ± Std)',
  box_plot:            'Box Plot',
  bar_chart:           'Bar Chart (Mean ± Std)',
  histogram:           'Histogram',
  convergence:         'Convergence Analysis',
  success_rate:        'Success Rate Over Time',
  improvement_speed:   'Learning Speed',
  stability:           'Training Stability',
  final_performance:   'Final Performance',
  heatmap:             'Performance Heatmap',
};

const CHART_DESCRIPTIONS = {
  training_curves:     'Rolling mean of episode length over training episodes for each selected model.',
  training_curves_std: 'Mean ± standard deviation across seeds. CSVs with matching names (differing only by seed) are grouped automatically.',
  box_plot:            'Box-and-whisker plot comparing the distribution of episode lengths across models.',
  bar_chart:           'Bar chart showing mean episode length with standard deviation error bars.',
  histogram:           'Overlaid histogram of episode length distributions for all selected models.',
  convergence:         'Convergence speed analysis: first episode crossing performance thresholds (50, 100, 150, 195 steps).',
  success_rate:        'Rolling percentage of episodes reaching the goal (≥195 steps). Shows how quickly each model learns to solve the task.',
  improvement_speed:   'Rate of improvement per episode (derivative of rolling mean). Positive = getting better, negative = regressing.',
  stability:           'Rolling standard deviation over time. Lower values indicate more consistent, stable training behavior.',
  final_performance:   'Mean episode length of the last N episodes (N = rolling window) for each model family, averaged across seeds.',
  heatmap:             'Color-coded matrix comparing all model families across key metrics: mean, median, max, std, success rate, and best window.',
};

function loadCsvList() {
  chartsLoaded = true;
  fetch('/api/csvs')
    .then(r => r.json())
    .then(csvs => {
      allCsvData  = csvs;
      allCsvPaths = csvs.map(c => c.path);
      // Build family map
      familyMap = {};
      for (const c of csvs) {
        if (!familyMap[c.family]) familyMap[c.family] = [];
        familyMap[c.family].push(c);
      }
      renderCsvSidebar(csvs);
    });
}

function renderCsvSidebar(csvs) {
  const list = document.getElementById('csvList');
  if (!csvs.length) {
    list.innerHTML = '<div class="no-models">No CSV data found.<br>Run experiments first.</div>';
    return;
  }
  // Group by "ep / category"
  const tree = {};
  for (const c of csvs) {
    if (!tree[c.group]) tree[c.group] = [];
    tree[c.group].push(c);
  }
  // Within each group, sub-group by family
  let html = '';
  for (const [group, items] of Object.entries(tree)) {
    html += `<div class="grp-hdr">${esc(group)}</div>`;
    // Collect families in this group
    const families = {};
    for (const c of items) {
      if (!families[c.family]) families[c.family] = [];
      families[c.family].push(c);
    }
    for (const [fam, members] of Object.entries(families)) {
      if (members.length > 1) {
        // Show a family group header with checkbox to select all seeds
        const famLabel = members[0].label.replace(/ S\d+/i, '').replace(/\s+/g, ' ');
        html += `<label class="grp-hdr-row" style="border-top:none;background:#f0f2ff;padding:5px 16px 2px;">
          <input type="checkbox" data-family="${esc(fam)}" onchange="onFamilyToggle(this)">
          <span>${esc(famLabel)} (${members.length} seeds)</span>
        </label>`;
      }
      for (const c of members) {
        html += `
        <label class="model-item">
          <input type="checkbox" value="${esc(c.path)}" data-family="${esc(c.family)}" onchange="onCsvToggle(this)">
          <span>${esc(c.label)}</span>
        </label>`;
      }
    }
  }
  list.innerHTML = html;
}

function onCsvToggle(cb) {
  cb.checked ? csvSelected.add(cb.value) : csvSelected.delete(cb.value);
  syncFamilyCheckbox(cb.dataset.family);
  updateCsvCount();
}

function onFamilyToggle(cb) {
  const fam = cb.dataset.family;
  const checked = cb.checked;
  document.querySelectorAll(`#csvList input[type=checkbox][value][data-family="${fam}"]`).forEach(c => {
    c.checked = checked;
    checked ? csvSelected.add(c.value) : csvSelected.delete(c.value);
  });
  updateCsvCount();
}

function syncFamilyCheckbox(fam) {
  if (!fam) return;
  const famCb = document.querySelector(`#csvList input[type=checkbox][data-family="${fam}"]:not([value])`);
  if (!famCb) return;
  const members = document.querySelectorAll(`#csvList input[type=checkbox][value][data-family="${fam}"]`);
  const allChecked = [...members].every(c => c.checked);
  const someChecked = [...members].some(c => c.checked);
  famCb.checked = allChecked;
  famCb.indeterminate = someChecked && !allChecked;
}

function updateCsvCount() {
  document.getElementById('csvCountTxt').textContent = csvSelected.size + ' selected';
}

function toggleAllCsvs() {
  const cbs = document.querySelectorAll('#csvList input[type=checkbox][value]');
  const allChecked = csvSelected.size === allCsvPaths.length;
  cbs.forEach(cb => {
    cb.checked = !allChecked;
    !allChecked ? csvSelected.add(cb.value) : csvSelected.delete(cb.value);
  });
  // Sync all family checkboxes
  document.querySelectorAll('#csvList input[type=checkbox][data-family]:not([value])').forEach(cb => {
    cb.checked = !allChecked;
    cb.indeterminate = false;
  });
  updateCsvCount();
}

function getSelectedChartTypes() {
  return [...document.querySelectorAll('#chartTypeList input[type=checkbox]:checked')].map(cb => cb.value);
}

function onChartTypeToggle() {
  const n = getSelectedChartTypes().length;
  document.getElementById('ctCountTxt').textContent = n + ' chart type' + (n !== 1 ? 's' : '') + ' selected';
}

function toggleAllChartTypes() {
  const cbs = document.querySelectorAll('#chartTypeList input[type=checkbox]');
  const allChecked = [...cbs].every(cb => cb.checked);
  cbs.forEach(cb => cb.checked = !allChecked);
  onChartTypeToggle();
}

function generateSelectedCharts() {
  if (!csvSelected.size) { alert('Select at least one CSV first.'); return; }
  const types = getSelectedChartTypes();
  if (!types.length) { alert('Select at least one chart type.'); return; }

  const win     = +document.getElementById('winSlider').value;
  const display = document.getElementById('chartDisplay');
  const status  = document.getElementById('chartStatus');
  const btn     = document.getElementById('chartBtn');

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Generating…';
  status.textContent = `Generating ${types.length} chart(s)…`;
  display.innerHTML = '<div class="hint"><div class="spinner" style="width:32px;height:32px;border-width:3px"></div><div>Generating ' + types.length + ' chart(s)…</div></div>';

  fetch('/api/multi-chart', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      csvs: [...csvSelected],
      chart_types: types,
      options: { window: win },
    }),
  })
  .then(r => r.json())
  .then(data => {
    btn.disabled = false;
    btn.innerHTML = '📊 Generate Charts';
    if (data.error) {
      status.textContent = 'Error: ' + data.error;
      display.innerHTML = `<div class="hint"><div class="hint-icon">⚠️</div><div>${esc(data.error)}</div></div>`;
      return;
    }
    const charts = data.charts || [];
    const ok = charts.filter(c => c.image);
    status.textContent = `Generated ${ok.length} / ${charts.length} chart(s) — ${csvSelected.size} CSV(s)`;

    if (charts.length === 1 && ok.length === 1) {
      // Single chart — display full-width without grid
      const c = ok[0];
      const title = CHART_TYPES_NICE[c.chart_type] || c.chart_type;
      display.innerHTML = `<div class="chart-card"><div class="chart-card-head">${esc(title)}</div><img src="data:image/png;base64,${c.image}" alt="${esc(title)}"></div>`;
    } else {
      // Multi-chart grid
      const useCols2 = ok.length >= 2;
      let html = `<div class="chart-grid${useCols2 ? ' cols-2' : ''}">`;
      for (const c of charts) {
        const title = CHART_TYPES_NICE[c.chart_type] || c.chart_type;
        html += `<div class="chart-card">`;
        html += `<div class="chart-card-head">${esc(title)}</div>`;
        if (c.image) {
          html += `<img src="data:image/png;base64,${c.image}" alt="${esc(title)}">`;
        } else {
          html += `<div class="chart-error">⚠️ ${esc(c.error || 'No data')}</div>`;
        }
        html += `</div>`;
      }
      html += '</div>';
      display.innerHTML = html;
    }
  })
  .catch(err => {
    btn.disabled = false;
    btn.innerHTML = '📊 Generate Charts';
    status.textContent = 'Error: ' + err.message;
    display.innerHTML = `<div class="hint"><div class="hint-icon">⚠️</div><div>Request failed</div></div>`;
  });
}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    # Allow ngrok / any reverse-proxy host header
    app.config["SERVER_NAME"] = None
    print("=" * 50)
    print("  CartPole HCRL Visualizer")
    print("  Local:  http://localhost:5000")
    print("  Expose: ngrok http 5000")
    print("=" * 50)
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)
