"""模块 1：把 HuggingFace 上的 Whisper-small 转换成 CT2 格式，并按指定精度量化。"""

from pathlib import Path

import yaml
from ctranslate2.converters import TransformersConverter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


def convert_one(model_name: str, output_dir: Path, quantization: str) -> None:
    if output_dir.exists():
        print(f"[skip] {output_dir} already exists")
        return

    print(f"[convert] {model_name} → {output_dir}  (quantization={quantization})")
    converter = TransformersConverter(model_name)
    converter.convert(
        output_dir=str(output_dir),
        quantization=quantization,
        force=False,
    )
    print(f"[done]   {output_dir}")


def main() -> None:
    cfg = load_config()
    model_name = cfg["model"]["name"]
    cache_dir = PROJECT_ROOT / cfg["model"]["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    short_name = model_name.split("/")[-1]  # openai/whisper-small → whisper-small

    storage_formats = sorted({run["storage"] for run in cfg["runs"]})
    for storage in storage_formats:
        output_dir = cache_dir / f"{short_name}-{storage}"
        convert_one(model_name, output_dir, storage)


if __name__ == "__main__":
    main()
