"""
run_all.py — Full automated pipeline for HCRL CartPole experiments.
====================================================================
Runs every step in order for a given episode count, fully automated
(oracle feedback, no human keyboard input required).

Usage:
    uv run python run_all.py --episodes 200
    uv run python run_all.py --episodes 500
    uv run python run_all.py --episodes 200 --analyze-only

Steps:
    1. Baseline training          (pure Q-Learning + reward shaping)
    2. HCRL timing experiment     (3 conditions: Early / Mid / Late, oracle)
    3. Compare models             (training curves + gameplay stats)
    4. Convergence analysis       (threshold crossing speeds)
"""

import argparse
import subprocess
import sys
import pathlib

sys.stdout.reconfigure(encoding="utf-8")

PYTHON = sys.executable


def run(cmd: list[str], step: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {step}")
    print("=" * 60)
    result = subprocess.run(
        [PYTHON] + cmd,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    if result.returncode != 0:
        print(f"  ERROR: Step failed (exit code {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200,
                        help="Total training episodes (default: 200)")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Evaluation episodes per model for gameplay comparison (default: 100)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip training, only run analysis on existing results")
    parser.add_argument("--skip-charts", action="store_true",
                        help="Skip compare_models and convergence_analysis chart steps")
    args = parser.parse_args()

    ep = args.episodes
    eval_ep = args.eval_episodes
    out_dir = pathlib.Path(f"experiment-results/ep{ep}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"  HCRL CARTPOLE FULL PIPELINE")
    print(f"  Training episodes : {ep}")
    print(f"  Eval episodes     : {eval_ep}")
    print(f"  Output directory  : {out_dir}")
    print(f"  Mode              : {'Analyze only' if args.analyze_only else 'Train + Analyze'}")
    if args.skip_charts:
        print(f"  Charts            : SKIPPED (--skip-charts)")
    print(f"{'#' * 60}")

    charts_flag = ["--skip-charts"] if args.skip_charts else []

    if not args.analyze_only:
        # Step 1: Baseline
        run(
            ["run.py", "--episodes", str(ep)],
            f"[1/4] Baseline Training ({ep} episodes)",
        )

        # Step 2: HCRL Timing Experiment (oracle, automated)
        run(
            ["feedback_timing_experiment.py", "--auto", "--episodes", str(ep)] + charts_flag,
            f"[2/4] HCRL Timing Experiment — Early / Mid / Late (oracle, {ep} episodes)",
        )

    else:
        print("\n[Skipping training steps — analyze-only mode]")

    if args.skip_charts:
        print("\n[Skipping chart steps — --skip-charts mode]")
    else:
        # Step 3: Compare Models
        run(
            ["compare_models.py", "--episodes", str(ep), "--eval-episodes", str(eval_ep)],
            f"[3/4] Compare Models ({eval_ep} eval episodes each)",
        )

        # Step 4: Convergence Analysis
        run(
            ["convergence_analysis.py", "--episodes", str(ep)],
            "[4/4] Convergence Speed Analysis",
        )

    print(f"\n{'#' * 60}")
    print(f"  ALL DONE — results saved to: {out_dir}")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
