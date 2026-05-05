"""模块 5：严谨的速度 benchmark — warmup + 多次取中位数 + 分桶 + 首段延迟。"""

import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import yaml
from faster_whisper import WhisperModel
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


def load_model(storage: str, compute_type: str, device: str, cpu_threads: int) -> WhisperModel:
    model_dir = PROJECT_ROOT / "models" / f"whisper-small-{storage}"
    return WhisperModel(str(model_dir), device=device, compute_type=compute_type, cpu_threads=cpu_threads)


def time_one_pass(model: WhisperModel, audio_path: Path, beam_size: int) -> tuple[float, float]:
    """跑一次推理, 返回 (首段延迟, 端到端总耗时), 单位秒。"""
    start = time.perf_counter()
    segments, _info = model.transcribe(str(audio_path), beam_size=beam_size, language="en", task="transcribe")
    seg_iter = iter(segments)
    first = next(seg_iter, None)
    ttfs = time.perf_counter() - start
    if first is not None:
        for _ in seg_iter:
            pass
    total = time.perf_counter() - start
    return ttfs, total


def warmup(model: WhisperModel, manifest: pd.DataFrame, n_warmup: int, beam_size: int) -> None:
    for row in tqdm(manifest.head(n_warmup).itertuples(index=False), total=n_warmup, desc="warmup"):
        time_one_pass(model, PROJECT_ROOT / row.audio_path, beam_size)


def benchmark(model: WhisperModel, manifest: pd.DataFrame, n_trials: int, beam_size: int) -> pd.DataFrame:
    rows = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="benchmark"):
        audio_path = PROJECT_ROOT / row.audio_path
        ttfs_list, total_list = [], []
        for _ in range(n_trials):
            ttfs, total = time_one_pass(model, audio_path, beam_size)
            ttfs_list.append(ttfs)
            total_list.append(total)
        rows.append({
            "id": row.id,
            "duration_sec": row.duration_sec,
            "ttfs_median": median(ttfs_list),
            "infer_median": median(total_list),
            "infer_p90": float(np.percentile(total_list, 90)),
            "rtf_median": median(total_list) / row.duration_sec,
        })
    return pd.DataFrame(rows)


def bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    """按音频时长分桶 (短 <5s / 中 5-10s / 长 >10s), 报每桶的 RTF 中位数。"""
    def bucket(d):
        if d < 5: return "short(<5s)"
        if d < 10: return "medium(5-10s)"
        return "long(>10s)"

    df = df.copy()
    df["bucket"] = df["duration_sec"].map(bucket)
    grouped = df.groupby("bucket", sort=False).agg(
        n=("id", "count"),
        rtf_median=("rtf_median", "median"),
        infer_median=("infer_median", "median"),
        ttfs_median=("ttfs_median", "median"),
    ).reset_index()
    return grouped


def main() -> None:
    cfg = load_config()
    runtime = cfg["runtime"]
    bench_cfg = cfg["benchmark"]

    manifest = pd.read_csv(PROJECT_ROOT / cfg["dataset"]["cache_dir"] / "librispeech_subset" / "manifest.csv")
    manifest = manifest.head(bench_cfg["num_samples"]).reset_index(drop=True)

    results_dir = PROJECT_ROOT / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for run in cfg["runs"]:
        name = run["name"]
        out_path = results_dir / f"benchmark-{name}.csv"
        if out_path.exists():
            print(f"[skip] {out_path.name} already exists")
            df = pd.read_csv(out_path)
        else:
            try:
                model = load_model(run["storage"], run["compute_type"], runtime["device"], runtime["cpu_threads"])
            except ValueError as e:
                print(f"[skip] {name}: {e}")
                continue
            print(f"[run]  {name}  (storage={run['storage']}, compute={run['compute_type']})")
            warmup(model, manifest, bench_cfg["n_warmup"], runtime["beam_size"])
            df = benchmark(model, manifest, bench_cfg["n_trials"], runtime["beam_size"])
            df.to_csv(out_path, index=False)

        # bucket summary per run
        buckets = bucket_summary(df)
        buckets.insert(0, "run", name)
        summary_rows.append(buckets)

        overall_rtf = df["infer_median"].sum() / df["duration_sec"].sum()
        print(f"[done] {name}: overall RTF = {overall_rtf:.3f}, median ttfs = {df['ttfs_median'].median():.3f}s")

    if summary_rows:
        summary = pd.concat(summary_rows, ignore_index=True)
        summary.to_csv(results_dir / "benchmark-summary.csv", index=False)
        print()
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
