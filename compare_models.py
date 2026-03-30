import pandas as pd
from matplotlib import pyplot as plt
import pathlib

def compare_models(experiment_dir: str = "experiment-results"):
    dir_path = pathlib.Path(experiment_dir)
    baseline_path = dir_path / "episode_history.csv"
    hcrl_path = dir_path / "hcrl_episode_history.csv"

    if not baseline_path.exists():
        print("Chưa có Dữ liệu Baseline. Hãy chạy lệnh 'uv run python run.py' trước để lấy mốc so sánh.")
        return
    
    if not hcrl_path.exists():
        print("Chưa có Dữ liệu HCRL. Hãy chạy lệnh 'uv run python train_hcrl.py' và thao tác bằng Tương tác con người trước.")
        return

    baseline_df = pd.read_csv(baseline_path, index_col="episode_index")
    hcrl_df = pd.read_csv(hcrl_path, index_col="episode_index")

    # Tính mượt (Moving Average) để dễ nhìn biểu đồ
    window_size = 20
    baseline_smooth = baseline_df["episode_length"].rolling(window=window_size, min_periods=1).mean()
    hcrl_smooth = hcrl_df["episode_length"].rolling(window=window_size, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    
    # Biểu đồ Baseline
    plt.plot(baseline_smooth.index, baseline_smooth, label="Baseline (Q-Learning gốc)", color="blue", linewidth=2)
    plt.scatter(baseline_df.index, baseline_df["episode_length"], color="blue", alpha=0.1, s=10)

    # Biểu đồ HCRL
    plt.plot(hcrl_smooth.index, hcrl_smooth, label="HCRL (Q-Learning + Nhập liệu con người)", color="red", linewidth=2)
    plt.scatter(hcrl_df.index, hcrl_df["episode_length"], color="red", alpha=0.1, s=10)

    plt.title("So sánh Hiệu suất Học: Baseline vs Human-Centered RL (Smoothed)")
    plt.xlabel("Episodes (Số vòng đời)")
    plt.ylabel("Episode Length (Số bước giữ thăng bằng)")
    plt.axhline(y=195, color='g', linestyle='--', label='Tiêu chuẩn Giải (Goal: 195)')
    
    plt.legend()
    plt.grid(True)
    
    output_path = dir_path / "comparison_chart.png"
    plt.savefig(output_path)
    print(f"Bản vẽ so sánh đã được lưu tại: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    compare_models()
