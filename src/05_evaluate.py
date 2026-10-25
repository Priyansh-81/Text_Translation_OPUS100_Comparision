import pandas as pd
import matplotlib.pyplot as plt
from src.common import RESULTS_DIR

def main():
    p = RESULTS_DIR / "metrics.csv"
    if not p.exists():
        raise SystemExit("No results/metrics.csv yet. Run model scripts first; this script will not invent results.")
    df = pd.read_csv(p)
    print("=" * 80)
    print("MODEL EVALUATION RESULTS:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    valid = df.dropna(subset=["bleu", "chrf"])
    if len(valid):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # Plot 1: BLEU and chrF
        x = range(len(valid))
        width = 0.35
        systems = valid["system"].tolist()
        
        bars1 = ax1.bar([i - width/2 for i in x], valid["bleu"], width, label="BLEU", color="#4285F4")
        bars2 = ax1.bar([i + width/2 for i in x], valid["chrf"], width, label="chrF", color="#34A853")
        
        ax1.set_ylabel("Score (0 - 100)", fontsize=11, fontweight="bold")
        ax1.set_title("Machine Translation Quality (BLEU & chrF)", fontsize=12, fontweight="bold")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(systems, rotation=15, ha="right", fontsize=10)
        ax1.legend(loc="upper left")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)
        
        for bar in bars1:
            h = bar.get_height()
            ax1.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            h = bar.get_height()
            ax1.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

        # Plot 2: Average Inference Time
        bars3 = ax2.bar(systems, valid["avg_inference_seconds"], color="#EA4335", width=0.5)
        ax2.set_ylabel("Average Seconds / Sample", fontsize=11, fontweight="bold")
        ax2.set_title("Inference Latency Comparison", fontsize=12, fontweight="bold")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(systems, rotation=15, ha="right", fontsize=10)
        ax2.grid(axis="y", linestyle="--", alpha=0.5)
        
        for bar in bars3:
            h = bar.get_height()
            ax2.annotate(f"{h:.2f}s", xy=(bar.get_x() + bar.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        
        plt.tight_layout()
        out_plot = RESULTS_DIR / "metric_comparison.png"
        plt.savefig(out_plot, dpi=200)
        print(f"Saved comparison plot -> {out_plot}")

if __name__ == "__main__":
    main()

