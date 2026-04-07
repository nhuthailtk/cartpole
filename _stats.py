import pathlib, numpy as np, pandas as pd

base = pathlib.Path("experiment-results/ep100")
results = {}

# Baseline
for s in [0,1,2]:
    p = base / f"baseline_s{s}_history.csv"
    if p.exists():
        df = pd.read_csv(p, index_col="episode_index")
        results.setdefault("baseline", []).extend(df["episode_length"].tolist())

# HCRL oracle
for s in [0,1,2]:
    p = base / "hcrl-oracle" / f"hcrl_oracle_s{s}_history.csv"
    if p.exists():
        df = pd.read_csv(p, index_col="episode_index")
        results.setdefault("hcrl_oracle", []).extend(df["episode_length"].tolist())

# VI-TAMER
for s in [0,1,2]:
    p = base / "vi-tamer" / f"vi_tamer_s{s}_history.csv"
    if p.exists():
        df = pd.read_csv(p, index_col="episode_index")
        results.setdefault("vi_tamer", []).extend(df["episode_length"].tolist())

# RLHF oracle
for s in [0,1,2]:
    p = base / "rlhf-oracle" / f"rlhf_oracle_s{s}_history.csv"
    if p.exists():
        df = pd.read_csv(p, index_col="episode_index")
        results.setdefault("rlhf_oracle", []).extend(df["episode_length"].tolist())

# RLHF ensemble
for s in [0,1,2]:
    p = base / "rlhf-ensemble" / f"rlhf_ensemble_s{s}_history.csv"
    if p.exists():
        df = pd.read_csv(p, index_col="episode_index")
        results.setdefault("rlhf_ensemble", []).extend(df["episode_length"].tolist())

# Timing experiment
timing_base = base / "timing-experiment"
for cond in ["early", "mid", "late", "full_feedback"]:
    for s in [0,1,2]:
        p = timing_base / f"{cond}_s{s}_history.csv"
        if p.exists():
            df = pd.read_csv(p, index_col="episode_index")
            results.setdefault(f"timing_{cond}", []).extend(df["episode_length"].tolist())

print("=== TRAINING CURVE SUMMARY (last 20 episodes per seed) ===")
for name, lens in results.items():
    arr = np.array(lens)
    # Per-seed last-20 mean
    per_seed = []
    chunk = len(arr) // 3 if len(arr) >= 300 else len(arr)
    for i in range(0, len(arr), max(1, chunk)):
        sl = arr[i:i+chunk]
        per_seed.append(np.mean(sl[-20:]))
    print(f"{name:30s}  mean_last20={np.mean(per_seed):.1f}  std={np.std(per_seed):.1f}  overall_mean={np.mean(arr):.1f}  overall_std={np.std(arr):.1f}")

print()
print("=== FULL EPISODE LENGTH STATS (all episodes, all seeds) ===")
for name, lens in results.items():
    arr = np.array(lens)
    print(f"{name:30s}  n={len(arr)}  mean={np.mean(arr):.1f}  std={np.std(arr):.1f}  min={arr.min()}  max={arr.max()}  pct195={np.mean(arr>=195)*100:.1f}%")
