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
import numpy as np
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

/* ── Two-column layout ── */
.layout { display: flex; flex: 1; overflow: hidden; }

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

<div class="layout">

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

    <div class="statusbar">
      <div><span class="dot" id="dot"></span><span id="statusTxt">Ready</span></div>
      <div class="pbar-wrap">
        <div class="pbar"><div class="pbar-fill" id="pFill" style="width:0"></div></div>
        <span id="pTxt">—</span>
      </div>
    </div>

  </main>
</div>

<script>
let es = null;
let selected = new Set();
let labelMap  = {};

// ── Bootstrap: load model list ──────────────────────────────────
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

  // Group by "ep / category"
  const tree = {};
  for (const m of models) {
    if (!tree[m.group]) tree[m.group] = [];
    tree[m.group].push(m);
  }

  let html = '';
  let idx  = 0;
  for (const [group, items] of Object.entries(tree)) {
    html += `<div class="grp-hdr">${esc(group)}</div>`;
    for (const m of items) {
      html += `
        <label class="model-item">
          <input type="checkbox" value="${esc(m.path)}" onchange="onToggle(this)">
          <span>${esc(m.label)}</span>
        </label>`;
      idx++;
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

// ── Play / Stop ─────────────────────────────────────────────────
function togglePlay() { es ? stopPlay() : startPlay(); }

function startPlay() {
  if (!selected.size) { alert('Select at least one model first.'); return; }

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

// ── SSE handler ──────────────────────────────────────────────────
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
    showResults(d.summary);
    stopPlay(false);
    setStatus(`Done — ${d.summary.length} model(s) evaluated`);
  }
}

// ── Build game cards ─────────────────────────────────────────────
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

// ── Results table ────────────────────────────────────────────────
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

// ── Helpers ──────────────────────────────────────────────────────
function setStatus(t) { document.getElementById('statusTxt').textContent = t; }
function setProgress(cur, total) {
  document.getElementById('pFill').style.width = total ? (cur / total * 100) + '%' : '0';
  document.getElementById('pTxt').textContent  = total ? `${cur} / ${total} ep` : '—';
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
