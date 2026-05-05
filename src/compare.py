"""模块 6：汇总 WER + 速度结果, 生成对比表和对比图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


def merge_results(results_dir: Path, runs: list[dict]) -> pd.DataFrame:
    """合并 WER summary 和 benchmark per-run 总体 RTF。"""
    rows = []
    wer_summary = pd.read_csv(results_dir / "wer-summary.csv") if (results_dir / "wer-summary.csv").exists() else None

    for run in runs:
        name = run["name"]
        bench_path = results_dir / f"benchmark-{name}.csv"
        if not bench_path.exists():
            continue
        bench = pd.read_csv(bench_path)

        wer_micro = None
        if wer_summary is not None:
            row = wer_summary[wer_summary["run"] == name]
            if not row.empty:
                wer_micro = float(row.iloc[0]["overall_wer_micro"])

        rows.append({
            "run": name,
            "storage": run["storage"],
            "compute_type": run["compute_type"],
            "wer_micro": wer_micro,
            "rtf_overall": bench["infer_median"].sum() / bench["duration_sec"].sum(),
            "rtf_short": bench[bench["duration_sec"] < 5]["rtf_median"].median(),
            "rtf_medium": bench[(bench["duration_sec"] >= 5) & (bench["duration_sec"] < 10)]["rtf_median"].median(),
            "rtf_long": bench[bench["duration_sec"] >= 10]["rtf_median"].median(),
            "ttfs_median": bench["ttfs_median"].median(),
        })
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # WER bar
    ax = axes[0]
    valid = df.dropna(subset=["wer_micro"])
    ax.bar(valid["run"], valid["wer_micro"] * 100, color="#3b7dd8")
    ax.set_ylabel("WER (%)")
    ax.set_title("Word Error Rate (lower is better)")
    for i, v in enumerate(valid["wer_micro"] * 100):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom")

    # RTF bar (overall + 3 buckets)
    ax = axes[1]
    x = range(len(df))
    width = 0.2
    for offset, col, label in [(-1.5, "rtf_short", "short"), (-0.5, "rtf_medium", "medium"),
                               (0.5, "rtf_long", "long"), (1.5, "rtf_overall", "overall")]:
        ax.bar([i + offset * width for i in x], df[col], width=width, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["run"])
    ax.set_ylabel("RTF (lower is faster)")
    ax.set_title("Real-Time Factor by audio duration")
    ax.legend(fontsize=8)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png", dpi=120)
    plt.close()


def main() -> None:
    cfg = load_config()
    results_dir = PROJECT_ROOT / cfg["output"]["results_dir"]

    df = merge_results(results_dir, cfg["runs"])
    if df.empty:
        print("[empty] no benchmark results found, run inference + benchmark first")
        return

    df.to_csv(results_dir / "final-comparison.csv", index=False)
    plot_comparison(df, results_dir)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(df.to_string(index=False))
    print()
    print(f"[saved] {results_dir / 'final-comparison.csv'}")
    print(f"[saved] {results_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
