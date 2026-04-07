"""
RLHF training with Ensemble Reward Model — Christiano et al. (2017) §2.2.

Adds all §2.2 algorithmic improvements over the baseline train_rlhf.py:

  1. Ensemble (K=3 models, bootstrapped training)          §2.2 bullet 1
  2. Uncertainty-based query selection (highest variance)  §2.2.4
  3. Reward normalisation (zero mean / unit std)           §2.2.1
  4. Oracle human-error noise (10% random responses)       §2.2.3

Two feedback modes:
  oracle (default) — simulated oracle with Boltzmann + 10% error noise.
  human  (--human) — real human watches the most uncertain clip pairs.

Usage
-----
    uv run python train_rlhf_ensemble.py                         # oracle
    uv run python train_rlhf_ensemble.py --human                 # real human
    uv run python train_rlhf_ensemble.py --episodes 200 --seed 0 --n-models 3
    uv run python train_rlhf_ensemble.py --human --episodes 100 --seed 0

Controls (--human mode)
-----------------------
  [A]   — Clip A was better
  [B]   — Clip B was better
  [S]   — Skip (tie / unsure)
  [Esc] — Quit early
"""

import argparse
import collections
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pygame

from cartpole import config as cfg
from cartpole.reward_model import EnsembleRewardModel, oracle_preference
from cartpole.train_utils import (
    collect_segment,
    make_agent,
    run_rl_episode,
    save_history_csv,
)

# ---------------------------------------------------------------------------
# Pygame UI constants (used only in --human mode)
# ---------------------------------------------------------------------------

_WIN_W   = 620
_WIN_H   = 460
_FRAME_W = 600
_FRAME_H = 400
_CLIP_FPS = 30

_COL_BG     = (30,  30,  40)
_COL_A      = (100, 160, 240)
_COL_B      = (240, 140,  60)
_COL_SKIP   = (160, 160, 160)
_COL_WHITE  = (255, 255, 255)
_COL_YELLOW = (255, 220,  60)


# ---------------------------------------------------------------------------
# Pygame UI helpers (--human mode only)
# ---------------------------------------------------------------------------

def _init_pygame():
    pygame.init()
    pygame.display.set_caption("RLHF Ensemble — Human Preference Labelling")
    screen = pygame.display.set_mode((_WIN_W, _WIN_H))
    font_l = pygame.font.SysFont("Arial", 28, bold=True)
    font_s = pygame.font.SysFont("Arial", 18)
    return screen, font_l, font_s


def _blit_frame(screen, frame: np.ndarray) -> None:
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    surf = pygame.transform.scale(surf, (_FRAME_W, _FRAME_H))
    screen.blit(surf, ((_WIN_W - _FRAME_W) // 2, 0))


def _draw_bar(screen, font, text: str, colour) -> None:
    bar_rect = pygame.Rect(0, _FRAME_H, _WIN_W, _WIN_H - _FRAME_H)
    pygame.draw.rect(screen, _COL_BG, bar_rect)
    label = font.render(text, True, colour)
    screen.blit(label, (_WIN_W // 2 - label.get_width() // 2, _FRAME_H + 14))


def _overlay_label(screen, font, text: str, colour) -> None:
    label = font.render(text, True, colour)
    bg = pygame.Surface((label.get_width() + 20, label.get_height() + 8), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 160))
    x = (_WIN_W - bg.get_width()) // 2
    screen.blit(bg, (x, 10))
    screen.blit(label, (x + 10, 14))


def _pump_quit() -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


def _wait_for_keypress(screen, font_s) -> bool:
    _draw_bar(screen, font_s, "Press any key to continue…", _COL_SKIP)
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                return event.key != pygame.K_ESCAPE


def _play_clip(screen, font_l, font_s, frames, label, colour, clock, fps=_CLIP_FPS) -> bool:
    for frame in frames:
        if _pump_quit():
            return False
        screen.fill(_COL_BG)
        _blit_frame(screen, frame)
        _overlay_label(screen, font_l, label, colour)
        _draw_bar(screen, font_s, f"Watching {label}…  ({fps} fps)", colour)
        pygame.display.flip()
        clock.tick(fps)
    return True


def _query_human(screen, font_l, font_s, clock, frames_a, frames_b,
                 pair_index, total_pairs, fps=_CLIP_FPS) -> float | None:
    """
    Show Clip A then Clip B, ask human which was better.
    Returns 1.0 (A), 0.0 (B), 0.5 (skip), or None (quit).
    """
    header = f"Pair {pair_index}/{total_pairs}"

    screen.fill(_COL_BG)
    _draw_bar(screen, font_s, f"{header}  |  Get ready for CLIP A…", _COL_A)
    pygame.display.flip()
    pygame.time.wait(600)
    if not _play_clip(screen, font_l, font_s, frames_a, "CLIP  A", _COL_A, clock, fps):
        return None

    screen.fill(_COL_BG)
    _blit_frame(screen, frames_a[-1])
    _overlay_label(screen, font_l, "CLIP  A  —  end", _COL_A)
    if not _wait_for_keypress(screen, font_s):
        return None

    screen.fill(_COL_BG)
    _draw_bar(screen, font_s, f"{header}  |  Get ready for CLIP B…", _COL_B)
    pygame.display.flip()
    pygame.time.wait(600)
    if not _play_clip(screen, font_l, font_s, frames_b, "CLIP  B", _COL_B, clock, fps):
        return None

    screen.fill(_COL_BG)
    _blit_frame(screen, frames_b[-1])
    _overlay_label(screen, font_l, "CLIP  B  —  end", _COL_B)
    if not _wait_for_keypress(screen, font_s):
        return None

    prompt = "Which was better?   [A]  Clip A     [B]  Clip B     [S]  Skip"
    screen.fill(_COL_BG)
    th_w, th_h = _FRAME_W // 2 - 10, _FRAME_H // 2
    for frame, x_off, lbl, col in [
        (frames_a[-1], 5,               "A", _COL_A),
        (frames_b[-1], _FRAME_W // 2 + 5, "B", _COL_B),
    ]:
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (th_w, th_h))
        screen.blit(surf, (x_off, 20))
        tag = font_l.render(f"CLIP {lbl}", True, col)
        screen.blit(tag, (x_off + th_w // 2 - tag.get_width() // 2, th_h + 28))
    _draw_bar(screen, font_s, prompt, _COL_YELLOW)
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_a:
                    print(f"  [{pair_index}/{total_pairs}] Human chose: A")
                    return 1.0
                if event.key == pygame.K_b:
                    print(f"  [{pair_index}/{total_pairs}] Human chose: B")
                    return 0.0
                if event.key == pygame.K_s:
                    print(f"  [{pair_index}/{total_pairs}] Human skipped")
                    return 0.5


def _collect_segment_with_frames(
    env: gym.Env,
    agent,
    seg_length: int,
    ensemble=None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    obs_list, frames = [], []
    obs, _ = env.reset()
    action = agent.begin_episode(obs)
    while len(obs_list) < seg_length:
        frame = env.render()
        obs_list.append(obs.copy())
        frames.append(frame)
        next_obs, env_reward, terminated, truncated, _ = env.step(action)
        if ensemble is not None:
            reward = ensemble.predict_normalised(next_obs)
        else:
            reward = float(env_reward)
        action = agent.act(next_obs, reward)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            action = agent.begin_episode(obs)
    return np.array(obs_list), frames


def _collect_human_preferences_uncertain(
    screen, font_l, font_s, clock,
    seg_buf_with_frames: list[tuple[np.ndarray, list]],
    ensemble: EnsembleRewardModel,
    n_pairs: int,
    rng: np.random.Generator,
    fps: int = _CLIP_FPS,
):
    """
    Select the most uncertain pairs (§2.2.4) and show them to the human.
    Returns (segs_a, segs_b, prefs) same as oracle version.
    """
    # Extract obs arrays for uncertainty selection
    obs_segs = [obs for obs, _ in seg_buf_with_frames]
    n_cand = n_pairs * cfg.ENSEMBLE_CANDIDATES_MULT
    selected_a, selected_b, _ = ensemble.select_uncertain_pairs(
        obs_segs, n_pairs, n_candidates=n_cand, rng=rng
    )

    # Map selected obs arrays back to (obs, frames) pairs
    obs_to_frames = {id(obs): frames for obs, frames in seg_buf_with_frames}

    segs_a, segs_b, prefs = [], [], []
    for k, (obs_a, obs_b) in enumerate(zip(selected_a, selected_b)):
        # Find matching frames — fall back to first entry if id lookup fails
        frames_a = obs_to_frames.get(id(obs_a), seg_buf_with_frames[0][1])
        frames_b = obs_to_frames.get(id(obs_b), seg_buf_with_frames[1][1])

        mu = _query_human(screen, font_l, font_s, clock, frames_a, frames_b,
                          pair_index=k + 1, total_pairs=n_pairs, fps=fps)
        if mu is None:
            print("  User quit during labelling.")
            return segs_a, segs_b, prefs
        segs_a.append(obs_a)
        segs_b.append(obs_b)
        prefs.append(mu)

    return segs_a, segs_b, prefs


# ---------------------------------------------------------------------------
# Oracle training (automated)
# ---------------------------------------------------------------------------

def train(total_episodes: int, seed: int, n_models: int) -> None:
    warmup_eps   = max(10, int(total_episodes * cfg.RLHF_WARMUP_FRACTION))
    remaining    = total_episodes - warmup_eps
    num_iter     = max(1, remaining // cfg.RLHF_EPISODES_PER_ITER)
    actual_total = warmup_eps + num_iter * cfg.RLHF_EPISODES_PER_ITER

    out = pathlib.Path(cfg.experiment_dir(total_episodes, "rlhf-ensemble"))
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  RLHF + Ensemble  —  {actual_total} episodes  seed={seed}")
    print(f"  Ensemble size     : K={n_models}")
    print(f"  Oracle error prob : {cfg.ENSEMBLE_ERROR_PROB}  (human noise §2.2.3)")
    print(f"  Query selection   : uncertainty-based ({cfg.ENSEMBLE_CANDIDATES_MULT}× candidates §2.2.4)")
    print(f"  Reward            : normalised (zero mean / unit std §2.2.1)")
    print(f"  Warm-up: {warmup_eps} eps  |  Iterations: {num_iter} × {cfg.RLHF_EPISODES_PER_ITER} eps")
    print(f"  Output: {out}")
    print("=" * 60)

    rng      = np.random.default_rng(seed)
    env      = gym.make("CartPole-v1")
    agent    = make_agent(rng)
    ensemble = EnsembleRewardModel(
        n_models=n_models, obs_dim=env.observation_space.shape[0], rng=rng
    )

    seg_buf: collections.deque[np.ndarray] = collections.deque(maxlen=cfg.RLHF_SEGMENT_BUFFER)
    episode_lengths: list[int]   = []
    rm_losses:       list[float] = []
    total_queries    = 0

    print("\n=== Phase 1: Warm-up ===")
    for ep in range(warmup_eps):
        episode_lengths.append(run_rl_episode(env, agent))
        if (ep + 1) % max(1, warmup_eps // 5) == 0:
            print(f"  ep {ep+1:4d}  avg(last 10)={np.mean(episode_lengths[-10:]):.1f}")

    print(f"\nCollecting {cfg.RLHF_WARMUP_SEGMENTS} warm-up segments…")
    for _ in range(cfg.RLHF_WARMUP_SEGMENTS):
        seg_buf.append(collect_segment(env, agent, rng))

    print("Bootstrapping ensemble reward model…")
    for _ in range(2):
        n_cand = cfg.RLHF_PAIRS_PER_ITER * cfg.ENSEMBLE_CANDIDATES_MULT
        segs_a, segs_b, _ = ensemble.select_uncertain_pairs(
            list(seg_buf), cfg.RLHF_PAIRS_PER_ITER, n_candidates=n_cand, rng=rng
        )
        prefs = [
            oracle_preference(a, b, rng, error_prob=cfg.ENSEMBLE_ERROR_PROB)
            for a, b in zip(segs_a, segs_b)
        ]
        total_queries += len(prefs)
        for _ in range(cfg.RLHF_RM_EPOCHS):
            loss = ensemble.train_on_preferences(segs_a, segs_b, prefs)
        rm_losses.append(loss)
    print(f"  Bootstrap loss: {loss:.4f}  |  queries: {total_queries}")

    print("\n=== Phase 2: RLHF loop (ensemble + uncertainty queries) ===")
    for it in range(1, num_iter + 1):
        iter_lengths = [
            run_rl_episode(env, agent, ensemble, normalise=True)
            for _ in range(cfg.RLHF_EPISODES_PER_ITER)
        ]
        episode_lengths.extend(iter_lengths)

        for _ in range(cfg.RLHF_SEGMENTS_PER_ITER):
            seg_buf.append(collect_segment(env, agent, rng, ensemble, normalise=True))

        n_cand = cfg.RLHF_PAIRS_PER_ITER * cfg.ENSEMBLE_CANDIDATES_MULT
        segs_a, segs_b, _ = ensemble.select_uncertain_pairs(
            list(seg_buf), cfg.RLHF_PAIRS_PER_ITER, n_candidates=n_cand, rng=rng
        )
        prefs = [
            oracle_preference(a, b, rng, error_prob=cfg.ENSEMBLE_ERROR_PROB)
            for a, b in zip(segs_a, segs_b)
        ]
        total_queries += len(prefs)

        for _ in range(cfg.RLHF_RM_EPOCHS):
            loss = ensemble.train_on_preferences(segs_a, segs_b, prefs)
        rm_losses.append(loss)

        if it % max(1, num_iter // 10) == 0 or it == 1:
            print(f"  iter {it:4d}/{num_iter}"
                  f"  avg_ep={np.mean(iter_lengths):6.1f}"
                  f"  rm_loss={loss:.4f}"
                  f"  queries={total_queries}")

    env.close()

    agent.save(out / f"rlhf_ensemble_s{seed}_model.npz")
    ensemble.save(out, prefix=f"rlhf_ensemble_s{seed}")
    save_history_csv(episode_lengths, out / f"rlhf_ensemble_s{seed}_history.csv")
    print(f"\nSaved to {out}/  |  Total oracle queries: {total_queries}")

    _plot(episode_lengths, rm_losses, warmup_eps, n_models, total_queries,
          f"RLHF Ensemble (K={n_models}) — {actual_total} eps, seed={seed}, {total_queries} queries",
          "purple", out / f"rlhf_ensemble_s{seed}_results.png")


# ---------------------------------------------------------------------------
# Human training (interactive)
# ---------------------------------------------------------------------------

def train_human(total_episodes: int, seed: int, n_models: int) -> None:
    """Train RLHF Ensemble with real human clip comparisons.

    Human is shown the most uncertain pairs (§2.2.4 uncertainty-based query
    selection) rather than random pairs, making each label maximally informative.
    """
    warmup_eps  = max(10, int(total_episodes * cfg.RLHF_WARMUP_FRACTION))
    remaining   = total_episodes - warmup_eps
    num_iter    = max(1, remaining // cfg.RLHF_EPISODES_PER_ITER)

    out = pathlib.Path(cfg.experiment_dir(total_episodes, "rlhf-ensemble-human"))
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  RLHF Ensemble (human)  —  {total_episodes} eps  seed={seed}  K={n_models}")
    print(f"  Query selection: uncertainty-based (most uncertain pairs shown)")
    print(f"  Reward: normalised (zero mean / unit std §2.2.1)")
    print(f"  Warm-up: {warmup_eps} eps  |  Iterations: {num_iter} × {cfg.RLHF_EPISODES_PER_ITER} eps")
    print(f"  Output: {out}")
    print("=" * 60)

    rng      = np.random.default_rng(seed)
    env_rgb  = gym.make("CartPole-v1", render_mode="rgb_array")
    env_rl   = gym.make("CartPole-v1")
    agent    = make_agent(rng)
    ensemble = EnsembleRewardModel(
        n_models=n_models, obs_dim=env_rgb.observation_space.shape[0], rng=rng
    )

    # Buffer stores (obs_array, frames) pairs so we can replay clips to the human
    seg_buf: collections.deque[tuple[np.ndarray, list]] = (
        collections.deque(maxlen=cfg.RLHF_SEGMENT_BUFFER)
    )
    episode_lengths: list[int]   = []
    rm_losses:       list[float] = []
    human_labels     = 0

    screen, font_l, font_s = _init_pygame()
    clock = pygame.time.Clock()

    # Intro screen
    screen.fill(_COL_BG)
    for text, fnt, col, y in [
        ("RLHF Ensemble — Human Labelling",      font_l, _COL_YELLOW,  70),
        (f"Ensemble K={n_models} — uncertainty-based pair selection", font_s, _COL_WHITE, 130),
        ("You will see the pairs the model is MOST uncertain about.", font_s, _COL_WHITE, 165),
        ("[A]  Clip A was better",  font_s, _COL_A,    220),
        ("[B]  Clip B was better",  font_s, _COL_B,    255),
        ("[S]  Skip / tie",         font_s, _COL_SKIP,  290),
        ("[Esc]  Quit early",       font_s, _COL_SKIP,  325),
        ("Press any key to start…", font_s, _COL_YELLOW, 400),
    ]:
        surf = fnt.render(text, True, col)
        screen.blit(surf, (_WIN_W // 2 - surf.get_width() // 2, y))
    pygame.display.flip()
    if not _wait_for_keypress(screen, font_s):
        pygame.quit()
        env_rgb.close()
        env_rl.close()
        return

    # Phase 1 — warm-up (no human needed)
    print("=== Phase 1: Warm-up ===")
    for ep in range(warmup_eps):
        episode_lengths.append(run_rl_episode(env_rl, agent))
        if (ep + 1) % max(1, warmup_eps // 5) == 0:
            print(f"  ep {ep+1:3d}  avg(10)={np.mean(episode_lengths[-10:]):.1f}")

    print(f"\nCollecting {cfg.RLHF_WARMUP_SEGMENTS} warm-up segments…")
    for _ in range(cfg.RLHF_WARMUP_SEGMENTS):
        seg = _collect_segment_with_frames(env_rgb, agent, cfg.RLHF_SEGMENT_LENGTH)
        seg_buf.append(seg)

    # Phase 2 — bootstrap with human labels on uncertain pairs
    print(f"\n=== Phase 2: Bootstrap — {cfg.RLHF_PAIRS_PER_ITER} preference pairs ===")
    # For bootstrap, ensemble is untrained so all pairs are equally uncertain — use random
    indices = list(range(len(seg_buf)))
    segs_a_obs, segs_b_obs, prefs = [], [], []
    for k in range(cfg.RLHF_PAIRS_PER_ITER):
        i, j = rng.choice(indices, size=2, replace=False)
        obs_a, frames_a = list(seg_buf)[i]
        obs_b, frames_b = list(seg_buf)[j]
        mu = _query_human(screen, font_l, font_s, clock, frames_a, frames_b,
                          pair_index=k + 1, total_pairs=cfg.RLHF_PAIRS_PER_ITER)
        if mu is None:
            print("  User quit during bootstrap.")
            break
        segs_a_obs.append(obs_a)
        segs_b_obs.append(obs_b)
        prefs.append(mu)

    human_labels += len(prefs)
    if len(prefs) >= 2:
        for _ in range(cfg.RLHF_RM_EPOCHS):
            loss = ensemble.train_on_preferences(segs_a_obs, segs_b_obs, prefs)
        rm_losses.append(loss)
        print(f"  Bootstrap done: {len(prefs)} labels, loss={loss:.4f}")

    # Phase 3 — RLHF loop with uncertainty-based query selection
    print("\n=== Phase 3: RLHF loop (ensemble + uncertainty-based human queries) ===")
    for iteration in range(1, num_iter + 1):
        if _pump_quit():
            print("  User quit.")
            break

        iter_lengths = [
            run_rl_episode(env_rl, agent, ensemble, normalise=True)
            for _ in range(cfg.RLHF_EPISODES_PER_ITER)
        ]
        episode_lengths.extend(iter_lengths)

        progress = (iteration - 1) / max(num_iter - 1, 1)
        iter_seg_len = int(25 + 75 * progress)
        iter_fps     = int(15 + 30 * progress)

        for _ in range(cfg.RLHF_SEGMENTS_PER_ITER):
            seg = _collect_segment_with_frames(env_rgb, agent, iter_seg_len, ensemble)
            seg_buf.append(seg)

        # Show human the most uncertain pairs (§2.2.4)
        segs_a_obs, segs_b_obs, prefs = _collect_human_preferences_uncertain(
            screen, font_l, font_s, clock,
            list(seg_buf), ensemble, cfg.RLHF_PAIRS_PER_ITER, rng, fps=iter_fps,
        )
        human_labels += len(prefs)

        if len(prefs) >= 2:
            for _ in range(cfg.RLHF_RM_EPOCHS):
                loss = ensemble.train_on_preferences(segs_a_obs, segs_b_obs, prefs)
            rm_losses.append(loss)
        elif rm_losses:
            loss = rm_losses[-1]

        print(f"  Iter {iteration:3d}/{num_iter}"
              f"  avg_ep={np.mean(iter_lengths):6.1f}"
              f"  rm_loss={loss:.4f}"
              f"  labels={human_labels}"
              f"  seg_len={iter_seg_len}")

        if len(prefs) == 0 and cfg.RLHF_PAIRS_PER_ITER > 0:
            break

    env_rgb.close()
    env_rl.close()
    pygame.quit()

    agent.save(out / f"rlhf_ensemble_human_s{seed}_model.npz")
    ensemble.save(out, prefix=f"rlhf_ensemble_human_s{seed}")
    save_history_csv(episode_lengths, out / f"rlhf_ensemble_human_s{seed}_history.csv")
    print(f"\nSaved to {out}/  |  Total human labels: {human_labels}")

    _plot(episode_lengths, rm_losses, warmup_eps, n_models, human_labels,
          f"RLHF Ensemble human (K={n_models}) — {total_episodes} eps, seed={seed}",
          "darkorchid", out / f"rlhf_ensemble_human_s{seed}_results.png")


# ---------------------------------------------------------------------------
# Shared plot helper
# ---------------------------------------------------------------------------

def _plot(
    episode_lengths: list[int],
    rm_losses: list[float],
    warmup_eps: int,
    n_models: int,
    total_queries: int,
    title: str,
    color: str,
    save_path: pathlib.Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13)

    ax = axes[0]
    lengths = np.array(episode_lengths)
    ax.plot(lengths, alpha=0.3, color=color)
    if len(lengths) >= 20:
        rm = np.convolve(lengths, np.ones(20) / 20, mode="valid")
        ax.plot(range(19, len(lengths)), rm, color=color, linewidth=2, label="Rolling mean (20)")
    ax.axvline(warmup_eps, color="orange", linestyle="--", linewidth=1, label="RLHF starts")
    ax.axhline(cfg.GOAL_LENGTH, color="gray", linestyle="--", alpha=0.5,
               label=f"Goal: {cfg.GOAL_LENGTH}")
    ax.set(xlabel="Episode", ylabel="Length", title="Policy performance")
    ax.legend(fontsize=8)
    ax.set_ylim(0, cfg.MAX_TIMESTEPS + 10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if rm_losses:
        ax.plot(rm_losses, color=color, marker="o", markersize=3)
    ax.set(xlabel="Reward model update", ylabel="Preference loss (avg ensemble)",
           title=f"Ensemble reward model (K={n_models})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RLHF with ensemble, uncertainty queries, reward normalisation"
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--n-models", type=int, default=cfg.ENSEMBLE_N_MODELS)
    parser.add_argument("--human",    action="store_true",
                        help="Use real human clip comparisons instead of simulated oracle")
    args = parser.parse_args()

    if args.human:
        train_human(args.episodes, args.seed, args.n_models)
    else:
        train(args.episodes, args.seed, args.n_models)
