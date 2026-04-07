"""
visual_compare.py — Watch multiple trained models play CartPole side by side.

Up to 6 models can be compared in a dynamic grid layout:
  1-2 models → 1×2 grid
  3-4 models → 2×2 grid
  5-6 models → 2×3 grid

Usage:
    uv run python visual_compare.py <model1.npz> <model2.npz> ... [options]

Options:
    --labels   Custom labels for each model (must match number of models)
    --episodes Number of episodes to run (default: 10)

Examples:
    # Compare 4 timing conditions
    uv run python visual_compare.py \\
        experiment-results/ep200/baseline_model.npz \\
        experiment-results/ep200/timing-experiment/early_model.npz \\
        experiment-results/ep200/timing-experiment/mid_model.npz \\
        experiment-results/ep200/timing-experiment/late_model.npz \\
        --labels Baseline "Early (0-20%)" "Mid (40-60%)" "Late (80-100%)"
"""

import argparse
import pathlib
import sys
import time

import gymnasium as gym
import numpy as np
import pygame

from cartpole.agents import QLearningAgent


# --- Color palette (up to 6 models) ---
PALETTE = [
    {"color": (60, 120, 220),  "bg": (220, 235, 255)},  # blue
    {"color": (40, 160, 70),   "bg": (225, 245, 225)},  # green
    {"color": (210, 120, 30),  "bg": (255, 240, 220)},  # orange
    {"color": (160, 50, 180),  "bg": (245, 225, 250)},  # purple
    {"color": (200, 40, 60),   "bg": (255, 225, 225)},  # red
    {"color": (30, 170, 180),  "bg": (220, 248, 250)},  # teal
]

# --- Layout constants ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 30

WHITE     = (255, 255, 255)
BLACK     = (0, 0, 0)
GRAY      = (200, 200, 200)
DARK_GRAY = (80, 80, 80)
FAIL_BG   = (255, 240, 240)


def grid_dims(n: int) -> tuple[int, int]:
    """Return (rows, cols) for n models (max 6)."""
    if n <= 2:
        return 1, n
    elif n <= 4:
        return 2, 2
    else:
        return 2, 3


def draw_cart_pole(surface, cx: int, cy: int, cell_w: int, cell_h: int, observation, color):
    """Draw a simple cart-pole visualization centered in a cell."""
    cart_x = observation[0]
    pole_angle = observation[2]

    usable_width = cell_w - 80
    px = cx + int((cart_x / 2.4) * (usable_width // 2))
    px = max(cx - usable_width // 2, min(px, cx + usable_width // 2))

    cart_w, cart_h = 50, 24
    cart_rect = pygame.Rect(px - cart_w // 2, cy - cart_h // 2, cart_w, cart_h)
    pygame.draw.rect(surface, color, cart_rect)
    pygame.draw.rect(surface, BLACK, cart_rect, 2)

    pole_len = 80
    pole_end_x = px + int(pole_len * np.sin(pole_angle))
    pole_end_y = cy - int(pole_len * np.cos(pole_angle))
    pygame.draw.line(surface, color, (px, cy - cart_h // 2), (pole_end_x, pole_end_y), 4)
    pygame.draw.circle(surface, color, (pole_end_x, pole_end_y), 6)

    track_y = cy + cart_h // 2 + 6
    pygame.draw.line(surface, GRAY,
                     (cx - usable_width // 2, track_y),
                     (cx + usable_width // 2, track_y), 2)


def run_visual_compare(model_paths: list[str], labels: list[str], num_episodes: int = 10):
    n = len(model_paths)
    rows, cols = grid_dims(n)
    cell_w = SCREEN_WIDTH // cols
    cell_h = SCREEN_HEIGHT // rows

    # Build model defs
    models = []
    for i, (path, label) in enumerate(zip(model_paths, labels)):
        models.append({
            "label": label,
            "path": path,
            **PALETTE[i],
        })

    # Load agents
    agents = []
    for m in models:
        p = pathlib.Path(m["path"])
        if not p.exists():
            print(f"Model not found: {p}")
            sys.exit(1)
        agents.append(QLearningAgent.load(p))

    # Create environments
    envs = [gym.make("CartPole-v1", max_episode_steps=None) for _ in models]

    # Init pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"CartPole: {n}-Model Visual Comparison")
    clock = pygame.time.Clock()
    font_title  = pygame.font.SysFont("consolas", 18, bold=True)
    font_medium = pygame.font.SysFont("consolas", 15)
    font_small  = pygame.font.SysFont("consolas", 12)

    all_steps: list[list[int]] = [[] for _ in models]

    def cell_origin(idx: int) -> tuple[int, int]:
        row, col = divmod(idx, cols)
        return col * cell_w, row * cell_h

    def cell_center(idx: int) -> tuple[int, int]:
        ox, oy = cell_origin(idx)
        return ox + cell_w // 2, oy + cell_h // 2

    try:
        for ep in range(num_episodes):
            observations = []
            actions = []
            dones = [False] * n
            steps = [0] * n

            for i, (agent, env) in enumerate(zip(agents, envs)):
                obs, _ = env.reset()
                observations.append(obs)
                actions.append(agent.begin_episode(obs))

            while not all(dones):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt

                for i in range(n):
                    if not dones[i]:
                        obs, _, term, trunc, _ = envs[i].step(actions[i])
                        observations[i] = obs
                        steps[i] += 1
                        if term or trunc:
                            dones[i] = True
                            all_steps[i].append(steps[i])
                        else:
                            actions[i] = agents[i].act(obs, reward=0.0)

                # Draw
                screen.fill(WHITE)

                for i, m in enumerate(models):
                    ox, oy = cell_origin(i)
                    cx, cy = cell_center(i)

                    bg = FAIL_BG if dones[i] else m["bg"]
                    pygame.draw.rect(screen, bg, (ox, oy, cell_w, cell_h))
                    pygame.draw.rect(screen, DARK_GRAY, (ox, oy, cell_w, cell_h), 1)

                    title = font_title.render(m["label"], True, m["color"])
                    screen.blit(title, (cx - title.get_width() // 2, oy + 10))

                    step_text = font_medium.render(f"Steps: {steps[i]}", True, BLACK)
                    screen.blit(step_text, (cx - step_text.get_width() // 2, oy + 32))

                    cart_y = oy + cell_h // 2 + 20
                    if not dones[i]:
                        draw_cart_pole(screen, cx, cart_y, cell_w, cell_h, observations[i], m["color"])
                    else:
                        fail = font_medium.render("FELL!", True, (220, 60, 60))
                        screen.blit(fail, (cx - fail.get_width() // 2, cart_y - 10))

                    if all_steps[i]:
                        avg  = np.mean(all_steps[i])
                        best = max(all_steps[i])
                        stat = font_small.render(f"Avg: {avg:.0f}  Best: {best}", True, DARK_GRAY)
                        screen.blit(stat, (cx - stat.get_width() // 2, oy + cell_h - 25))

                ep_text = font_medium.render(
                    f"Episode {ep + 1} / {num_episodes}", True, DARK_GRAY,
                )
                screen.blit(ep_text, (
                    SCREEN_WIDTH // 2 - ep_text.get_width() // 2,
                    SCREEN_HEIGHT // 2 - ep_text.get_height() // 2,
                ))

                pygame.display.flip()
                clock.tick(FPS)

            time.sleep(0.3)

        # Final summary table
        col_w = 12
        header_w = 22 + col_w * n
        print("\n" + "=" * header_w)
        header = f"{'':>22}" + "".join(f"{m['label']:>{col_w}}" for m in models)
        print(header)
        print("=" * header_w)
        for stat_name, stat_fn in [
            ("Mean steps",   lambda s: f"{np.mean(s):.1f}"),
            ("Median steps", lambda s: f"{np.median(s):.1f}"),
            ("Best",         lambda s: f"{max(s)}"),
            ("Worst",        lambda s: f"{min(s)}"),
        ]:
            row = f"{stat_name:>22}" + "".join(f"{stat_fn(s):>{col_w}}" for s in all_steps)
            print(row)
        print("=" * header_w)

        print("\nClose the window or press Esc to exit.")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    waiting = False
            clock.tick(10)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        for env in envs:
            env.close()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Watch multiple trained CartPole models side by side (max 6).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("models", nargs="+", metavar="model.npz",
                        help="Paths to model .npz files (1–6 models)")
    parser.add_argument("--labels", nargs="+", metavar="LABEL",
                        help="Display labels for each model (must match number of models)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to play (default: 10)")
    args = parser.parse_args()

    PREFIX = "experiment-results/"
    models = [p if p.startswith(PREFIX) else PREFIX + p for p in args.models]

    if len(models) > 6:
        parser.error("Maximum 6 models supported.")

    labels = args.labels or [pathlib.Path(p).stem for p in models]

    if len(labels) != len(models):
        parser.error(f"--labels count ({len(labels)}) must match model count ({len(models)}).")

    run_visual_compare(models, labels, args.episodes)


if __name__ == "__main__":
    main()
