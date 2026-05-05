"""模块 2：从 HuggingFace 下载 LibriSpeech test-clean 子集，存为本地 WAV + manifest.csv。"""

from itertools import islice
from pathlib import Path

import pandas as pd
import soundfile as sf
import yaml
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


def stream_librispeech(num_samples: int):
    """流式拉取 LibriSpeech test-clean 的前 num_samples 条样本。"""
    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="test",
        streaming=True,
        trust_remote_code=True,
    )
    return islice(ds, num_samples)


def save_subset(samples_iter, output_dir: Path, num_samples: int) -> pd.DataFrame:
    """把流式样本逐条落盘为 WAV，同时返回 manifest DataFrame。"""
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample in tqdm(samples_iter, total=num_samples, desc="downloading"):
        sample_id = sample["id"]
        audio = sample["audio"]
        wav_path = audio_dir / f"{sample_id}.wav"

        sf.write(wav_path, audio["array"], audio["sampling_rate"])

        rows.append({
            "id": sample_id,
            "audio_path": str(wav_path.relative_to(PROJECT_ROOT)),
            "reference": sample["text"],
            "duration_sec": len(audio["array"]) / audio["sampling_rate"],
        })

    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    num_samples = cfg["dataset"]["num_samples"]
    output_dir = PROJECT_ROOT / cfg["dataset"]["cache_dir"] / "librispeech_subset"
    manifest_path = output_dir / "manifest.csv"

    if manifest_path.exists():
        print(f"[skip] {manifest_path} already exists ({len(pd.read_csv(manifest_path))} rows)")
        return

    samples = stream_librispeech(num_samples)
    manifest = save_subset(samples, output_dir, num_samples)
    manifest.to_csv(manifest_path, index=False)
    print(f"[done] wrote {len(manifest)} samples to {output_dir}")
    print(f"       total audio duration: {manifest['duration_sec'].sum():.1f} sec")


if __name__ == "__main__":
    main()
