"""模块 3：加载某精度的 Whisper 模型，对 manifest 中的音频跑推理，输出预测 CSV。"""

import time
from pathlib import Path

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
    return WhisperModel(
        str(model_dir),
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
    )


def transcribe_one(model: WhisperModel, audio_path: Path, beam_size: int) -> tuple[str, float]:
    start = time.perf_counter()
    segments, _info = model.transcribe(
        str(audio_path),
        beam_size=beam_size,
        language="en",
        task="transcribe",
    )
    text = " ".join(seg.text.strip() for seg in segments)
    elapsed = time.perf_counter() - start
    return text, elapsed


def run_inference(model: WhisperModel, manifest: pd.DataFrame, beam_size: int) -> pd.DataFrame:
    rows = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="inference"):
        audio_path = PROJECT_ROOT / row.audio_path
        prediction, elapsed = transcribe_one(model, audio_path, beam_size)
        rows.append({
            "id": row.id,
            "reference": row.reference,
            "prediction": prediction,
            "duration_sec": row.duration_sec,
            "infer_sec": elapsed,
        })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    runtime = cfg["runtime"]
    manifest_path = PROJECT_ROOT / cfg["dataset"]["cache_dir"] / "librispeech_subset" / "manifest.csv"
    manifest = pd.read_csv(manifest_path)

    results_dir = PROJECT_ROOT / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    for run in cfg["runs"]:
        name = run["name"]
        out_path = results_dir / f"predictions-{name}.csv"

        if out_path.exists():
            print(f"[skip] {out_path.name} already exists")
            continue

        try:
            model = load_model(
                storage=run["storage"],
                compute_type=run["compute_type"],
                device=runtime["device"],
                cpu_threads=runtime["cpu_threads"],
            )
        except ValueError as e:
            print(f"[skip] {name}: {e}")
            continue

        print(f"[run]  {name}  (storage={run['storage']}, compute={run['compute_type']}, device={runtime['device']})")
        df = run_inference(model, manifest, runtime["beam_size"])
        df.to_csv(out_path, index=False)
        print(f"[done] {name}: {len(df)} predictions, mean infer {df['infer_sec'].mean():.2f}s, → {out_path.name}")


if __name__ == "__main__":
    main()
