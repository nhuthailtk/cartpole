"""
RLHF training with real human preferences — Christiano et al. (2017).

Interactive version of train_rlhf.py.  Instead of a simulated oracle, a real
human watches pairs of CartPole clips and presses a key to say which looks
better.  The reward model learns from those labels and drives Q-learning.

Controls during preference labelling
-------------------------------------
  [A]    — Clip A was better
  [B]    — Clip B was better
  [S]    — Skip (tie / unsure)
  [Esc]  — Quit training early

Pipeline
--------
1. Warm-up: run episodes with env reward; collect initial segments.
2. Bootstrap: show human WARMUP_PAIRS clip pairs to seed the reward model.
3. Main loop (NUM_ITERATIONS):
   a. Run EPISODES_PER_ITER episodes using reward model as reward signal.
   b. Collect SEGMENTS_PER_ITER new segments.
   c. Show human PAIRS_PER_ITER clip pairs; record preferences.
   d. Train reward model on all accumulated preferences.
4. Save model + plot results.

Usage
-----
    uv run python train_rlhf_human.py
"""

import collections
import pathlib
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pygame

from cartpole.agents import QLearningAgent
from cartpole.reward_model import RewardModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = pathlib.Path("experiment-results")

SEED = 2

# Segments
SEGMENT_LENGTH      = 50    # timesteps per clip
WARMUP_SEGMENTS     = 20    # segments collected before any labelling
SEGMENTS_PER_ITER   = 4     # new segments per iteration
SEGMENT_BUFFER_SIZE = 300

# Human labelling  (kept low to avoid fatigue)
WARMUP_PAIRS        = 10    # pairs shown during bootstrap
PAIRS_PER_ITER      = 4     # pairs shown each iteration
REWARD_MODEL_EPOCHS = 50    # gradient steps after each labelling round

# RL
WARMUP_EPISODES         = 64
NUM_ITERATIONS          = 17   # 64 + 17×8 = 200 episodes total
EPISODES_PER_ITER       = 8
MAX_TIMESTEPS           = 200

# Agent
AGENT_LR                = 0.2
AGENT_DISCOUNT          = 1.0
AGENT_EXPLORATION       = 0.5
AGENT_EXPLORATION_DECAY = 0.99

# Reward model
REWARD_MODEL_LR     = 3e-4
REWARD_MODEL_HIDDEN = 64

# Display
CLIP_FPS   = 30          # replay speed when showing clips to human
WIN_W      = 620         # pygame window width
WIN_H      = 460         # pygame window height  (400 frame + 60 text bar)
FRAME_W    = 600         # CartPole rgb_array width
FRAME_H    = 400         # CartPole rgb_array height

# Colours
COL_BG      = (30,  30,  40)
COL_A       = (100, 160, 240)   # blue  — Clip A
COL_B       = (240, 140,  60)   # amber — Clip B
COL_SKIP    = (160, 160, 160)
COL_WHITE   = (255, 255, 255)
COL_YELLOW  = (255, 220,  60)


# ---------------------------------------------------------------------------
# Pygame helpers
# ---------------------------------------------------------------------------

def _init_pygame() -> tuple[pygame.Surface, pygame.font.Font, pygame.font.Font]:
    pygame.init()
    pygame.display.set_caption("RLHF — Human Preference Labelling")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    font_large = pygame.font.SysFont("Arial", 28, bold=True)
    font_small = pygame.font.SysFont("Arial", 18)
    return screen, font_large, font_small


def _blit_frame(screen: pygame.Surface, frame: np.ndarray) -> None:
    """Blit a (H, W, 3) uint8 rgb_array onto the screen, centred."""
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    surf = pygame.transform.scale(surf, (FRAME_W, FRAME_H))
    x = (WIN_W - FRAME_W) // 2
    screen.blit(surf, (x, 0))


def _draw_bar(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    colour: tuple[int, int, int],
) -> None:
    """Draw the bottom text bar."""
    bar_rect = pygame.Rect(0, FRAME_H, WIN_W, WIN_H - FRAME_H)
    pygame.draw.rect(screen, COL_BG, bar_rect)
    label = font.render(text, True, colour)
    screen.blit(label, (WIN_W // 2 - label.get_width() // 2, FRAME_H + 14))


def _overlay_label(
    screen: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    colour: tuple[int, int, int],
) -> None:
    """Semi-transparent label at the top of the frame area."""
    label = font.render(text, True, colour)
    bg = pygame.Surface((label.get_width() + 20, label.get_height() + 8), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 160))
    x = (WIN_W - bg.get_width()) // 2
    screen.blit(bg, (x, 10))
    screen.blit(label, (x + 10, 14))


def _pump_quit() -> bool:
    """Return True if the user closed the window or pressed Esc."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


def _wait_for_keypress(screen: pygame.Surface, font_s: pygame.font.Font) -> bool:
    """
    Block until user presses any key.
    Returns False if user quit / pressed Esc.
    """
    _draw_bar(screen, font_s, "Press any key to continue…", COL_SKIP)
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                return True


# ---------------------------------------------------------------------------
# Clip playback
# ---------------------------------------------------------------------------

def _play_clip(
    screen: pygame.Surface,
    font_l: pygame.font.Font,
    font_s: pygame.font.Font,
    frames: list[np.ndarray],
    label: str,
    colour: tuple[int, int, int],
    clock: pygame.time.Clock,
    fps: int = CLIP_FPS,
) -> bool:
    """
    Replay a list of frames at `fps`.
    Returns False if user quit mid-playback.
    """
    for frame in frames:
        if _pump_quit():
            return False
        screen.fill(COL_BG)
        _blit_frame(screen, frame)
        _overlay_label(screen, font_l, label, colour)
        _draw_bar(screen, font_s, f"Watching {label}…  ({fps} fps)", colour)
        pygame.display.flip()
        clock.tick(fps)
    return True


# ---------------------------------------------------------------------------
# Human preference query
# ---------------------------------------------------------------------------

def query_human(
    screen: pygame.Surface,
    font_l: pygame.font.Font,
    font_s: pygame.font.Font,
    clock: pygame.time.Clock,
    frames_a: list[np.ndarray],
    frames_b: list[np.ndarray],
    pair_index: int,
    total_pairs: int,
    fps: int = CLIP_FPS,
) -> float | None:
    """
    Show Clip A then Clip B, then ask the human which was better.

    Returns
    -------
    1.0   — human preferred A
    0.0   — human preferred B
    0.5   — skip / tie
    None  — user quit (Esc / window close)
    """
    header = f"Pair {pair_index}/{total_pairs}"

    # --- Play Clip A ---
    screen.fill(COL_BG)
    _draw_bar(screen, font_s, f"{header}  |  Get ready for CLIP A…", COL_A)
    pygame.display.flip()
    pygame.time.wait(600)

    if not _play_clip(screen, font_l, font_s, frames_a, "CLIP  A", COL_A, clock, fps):
        return None

    # Brief pause between clips
    screen.fill(COL_BG)
    _blit_frame(screen, frames_a[-1])
    _overlay_label(screen, font_l, "CLIP  A  —  end", COL_A)
    if not _wait_for_keypress(screen, font_s):
        return None

    # --- Play Clip B ---
    screen.fill(COL_BG)
    _draw_bar(screen, font_s, f"{header}  |  Get ready for CLIP B…", COL_B)
    pygame.display.flip()
    pygame.time.wait(600)

    if not _play_clip(screen, font_l, font_s, frames_b, "CLIP  B", COL_B, clock, fps):
        return None

    screen.fill(COL_BG)
    _blit_frame(screen, frames_b[-1])
    _overlay_label(screen, font_l, "CLIP  B  —  end", COL_B)
    if not _wait_for_keypress(screen, font_s):
        return None

    # --- Ask for preference ---
    prompt = "Which was better?   [A]  Clip A     [B]  Clip B     [S]  Skip"
    _draw_bar(screen, font_s, prompt, COL_YELLOW)

    # Show last frame of each clip side by side as thumbnails
    screen.fill(COL_BG)
    th_w, th_h = FRAME_W // 2 - 10, FRAME_H // 2
    for frame, x_off, lbl, col in [
        (frames_a[-1], 5,             "A", COL_A),
        (frames_b[-1], FRAME_W // 2 + 5, "B", COL_B),
    ]:
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (th_w, th_h))
        screen.blit(surf, (x_off, 20))
        tag = font_l.render(f"CLIP {lbl}", True, col)
        screen.blit(tag, (x_off + th_w // 2 - tag.get_width() // 2, th_h + 28))

    _draw_bar(screen, font_s, prompt, COL_YELLOW)
    pygame.display.flip()

    # Wait for A / B / S
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
                    print(f"  [{pair_index}/{total_pairs}] Human skipped (tie)")
                    return 0.5


# ---------------------------------------------------------------------------
# Segment collection
# ---------------------------------------------------------------------------

def collect_segment_with_frames(
    env: gym.Env,
    agent: QLearningAgent,
    seg_length: int,
    reward_model: RewardModel | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Run the agent for `seg_length` steps.

    Returns
    -------
    obs_array : (seg_length, obs_dim)
    frames    : list of (H, W, 3) uint8 arrays, one per step
    """
    obs_list: list[np.ndarray]   = []
    frames:   list[np.ndarray]   = []

    obs, _ = env.reset()
    action = agent.begin_episode(obs)

    while len(obs_list) < seg_length:
        frame = env.render()
        obs_list.append(obs.copy())
        frames.append(frame)

        next_obs, env_reward, terminated, truncated, _ = env.step(action)
        reward = reward_model.predict(next_obs) if reward_model is not None else env_reward
        action = agent.act(next_obs, reward)
        obs = next_obs

        if terminated or truncated:
            obs, _ = env.reset()
            action = agent.begin_episode(obs)

    return np.array(obs_list), frames


def run_episode(
    env: gym.Env,
    agent: QLearningAgent,
    reward_model: RewardModel | None = None,
) -> int:
    """Run one full episode; return length."""
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


# ---------------------------------------------------------------------------
# Labelling round
# ---------------------------------------------------------------------------

def collect_preferences(
    screen: pygame.Surface,
    font_l: pygame.font.Font,
    font_s: pygame.font.Font,
    clock: pygame.time.Clock,
    segment_buffer: list[tuple[np.ndarray, list[np.ndarray]]],
    n_pairs: int,
    rng: np.random.Generator,
    fps: int = CLIP_FPS,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """
    Sample `n_pairs` segment pairs, show them to the human, collect labels.
    Skips segments skipped by the user (returns however many were labelled).
    """
    segs_a, segs_b, prefs = [], [], []
    indices = list(range(len(segment_buffer)))

    for k in range(n_pairs):
        i, j = rng.choice(indices, size=2, replace=False)
        obs_a, frames_a = segment_buffer[i]
        obs_b, frames_b = segment_buffer[j]

        mu = query_human(screen, font_l, font_s, clock, frames_a, frames_b,
                         pair_index=k + 1, total_pairs=n_pairs, fps=fps)
        if mu is None:          # user quit
            print("  User quit during labelling — stopping early.")
            return segs_a, segs_b, prefs

        segs_a.append(obs_a)
        segs_b.append(obs_b)
        prefs.append(mu)

    return segs_a, segs_b, prefs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng   = np.random.default_rng(SEED)
    # rgb_array env for segment collection (frames for human display)
    env   = gym.make("CartPole-v1", render_mode="rgb_array")
    # headless env for RL episode training
    env_rl = gym.make("CartPole-v1")

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

    # segment buffer stores (obs_array, frames_list) pairs
    segment_buffer: collections.deque[tuple[np.ndarray, list[np.ndarray]]] = (
        collections.deque(maxlen=SEGMENT_BUFFER_SIZE)
    )

    episode_lengths: list[int]   = []
    rm_losses:       list[float] = []
    human_labels:    int         = 0

    screen, font_l, font_s = _init_pygame()
    clock = pygame.time.Clock()

    # ------------------------------------------------------------------ #
    # Intro screen                                                         #
    # ------------------------------------------------------------------ #
    screen.fill(COL_BG)
    lines = [
        ("RLHF  —  Human Preference Labelling",  font_l, COL_YELLOW,  80),
        ("You will watch pairs of CartPole clips", font_s, COL_WHITE, 150),
        ("and press a key to say which looks better.", font_s, COL_WHITE, 185),
        ("[A]  Clip A was better",  font_s, COL_A,     240),
        ("[B]  Clip B was better",  font_s, COL_B,     275),
        ("[S]  Skip / tie",         font_s, COL_SKIP,  310),
        ("[Esc]  Quit early",       font_s, COL_SKIP,  345),
        ("Press any key to start…", font_s, COL_YELLOW, 410),
    ]
    for text, fnt, col, y in lines:
        surf = fnt.render(text, True, col)
        screen.blit(surf, (WIN_W // 2 - surf.get_width() // 2, y))
    pygame.display.flip()
    if not _wait_for_keypress(screen, font_s):
        pygame.quit()
        return

    # ------------------------------------------------------------------ #
    # Phase 1 — Warm-up: train on env reward, collect initial segments    #
    # ------------------------------------------------------------------ #
    print("=== Phase 1: Warm-up ===")
    for ep in range(WARMUP_EPISODES):
        length = run_episode(env_rl, agent, reward_model=None)
        episode_lengths.append(length)
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1:3d}  avg(10)={np.mean(episode_lengths[-10:]):.1f}")

    print(f"\nCollecting {WARMUP_SEGMENTS} warm-up segments…")
    for _ in range(WARMUP_SEGMENTS):
        seg = collect_segment_with_frames(env, agent, SEGMENT_LENGTH, reward_model=None)
        segment_buffer.append(seg)

    # ------------------------------------------------------------------ #
    # Phase 2 — Bootstrap reward model from initial human labels          #
    # ------------------------------------------------------------------ #
    print(f"\n=== Phase 2: Bootstrap — {WARMUP_PAIRS} preference pairs ===")
    segs_a, segs_b, prefs = collect_preferences(
        screen, font_l, font_s, clock,
        list(segment_buffer), WARMUP_PAIRS, rng,
    )
    human_labels += len(prefs)

    if len(prefs) >= 2:
        for _ in range(REWARD_MODEL_EPOCHS):
            loss = reward_model.train_on_preferences(segs_a, segs_b, prefs)
        rm_losses.append(loss)
        print(f"  Bootstrap done: {len(prefs)} labels, loss={loss:.4f}")
    else:
        print("  Not enough labels to train — using env reward only.")

    # ------------------------------------------------------------------ #
    # Phase 3 — RLHF main loop                                            #
    # ------------------------------------------------------------------ #
    print("\n=== Phase 3: RLHF loop ===")
    for iteration in range(1, NUM_ITERATIONS + 1):

        # Check for quit (window closed between iterations)
        if _pump_quit():
            print("  User quit — saving and stopping.")
            break

        # 3a. RL episodes with reward model
        iter_lengths = []
        for _ in range(EPISODES_PER_ITER):
            length = run_episode(env_rl, agent, reward_model=reward_model)
            iter_lengths.append(length)
        episode_lengths.extend(iter_lengths)

        # Clip length and FPS scale linearly with iteration:
        #   iter 1  → seg_len=25,  fps=15
        #   iter 59 → seg_len=100, fps=45
        progress = (iteration - 1) / max(NUM_ITERATIONS - 1, 1)   # 0.0 → 1.0
        iter_seg_len = int(25 + 75 * progress)    # 25 → 100
        iter_fps     = int(15 + 30 * progress)    # 15 → 45

        # 3b. Collect new segments
        for _ in range(SEGMENTS_PER_ITER):
            seg = collect_segment_with_frames(env, agent, iter_seg_len, reward_model)
            segment_buffer.append(seg)

        # 3c. Human labelling round  (pass iter_fps into _play_clip via clock)
        clock_fps = iter_fps
        segs_a, segs_b, prefs = collect_preferences(
            screen, font_l, font_s, clock,
            list(segment_buffer), PAIRS_PER_ITER, rng,
            fps=clock_fps,
        )
        human_labels += len(prefs)

        # 3d. Train reward model
        if len(prefs) >= 2:
            for _ in range(REWARD_MODEL_EPOCHS):
                loss = reward_model.train_on_preferences(segs_a, segs_b, prefs)
            rm_losses.append(loss)
        elif rm_losses:
            loss = rm_losses[-1]   # carry forward if user skipped all

        avg_len = np.mean(iter_lengths)
        print(
            f"  Iter {iteration:3d}/{NUM_ITERATIONS}"
            f"  avg_ep={avg_len:6.1f}"
            f"  rm_loss={loss:.4f}"
            f"  labels_total={human_labels}"
            f"  seg_len={iter_seg_len}  fps={iter_fps}"
        )

        # User may have quit during labelling (collect_preferences returns early)
        if len(prefs) == 0 and PAIRS_PER_ITER > 0:
            break

    env.close()
    env_rl.close()
    pygame.quit()

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    reward_model.save(OUTPUT_DIR / "reward_model_human.npz")
    agent.save(OUTPUT_DIR / "rlhf_human_model.npz")
    print(f"Agent saved to {OUTPUT_DIR / 'rlhf_human_model.npz'}")

    # ------------------------------------------------------------------ #
    # Plot                                                                 #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"RLHF (Human) on CartPole — {human_labels} preference labels",
        fontsize=13,
    )

    ax = axes[0]
    lengths = np.array(episode_lengths)
    ax.plot(lengths, alpha=0.35, color="steelblue")
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

    ax = axes[1]
    ax.plot(rm_losses, color="tomato", marker="o", markersize=4)
    ax.set_xlabel("Reward model update")
    ax.set_ylabel("Preference cross-entropy loss")
    ax.set_title(f"Reward model  ({human_labels} human labels)")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "rlhf_human_results.png"
    plt.savefig(plot_path, dpi=120)
    print(f"\nPlot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    train()
