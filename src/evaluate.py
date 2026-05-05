"""模块 4：用 Whisper 的 EnglishTextNormalizer 归一化后，对每个 run 算 WER。"""

from pathlib import Path

import jiwer
import pandas as pd
import yaml
from transformers import WhisperTokenizer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


def build_normalizer(model_name: str) -> EnglishTextNormalizer:
    tokenizer = WhisperTokenizer.from_pretrained(model_name)
    return EnglishTextNormalizer(tokenizer.english_spelling_normalizer)


def compute_wer(predictions: pd.DataFrame, normalizer: EnglishTextNormalizer) -> tuple[float, pd.DataFrame]:
    df = predictions.copy()
    df["ref_norm"] = df["reference"].map(normalizer)
    df["hyp_norm"] = df["prediction"].map(normalizer)

    # 过滤归一化后变空的行 (jiwer 不接受空 reference)
    valid = df[df["ref_norm"].str.strip().astype(bool)].copy()

    valid["wer"] = [
        jiwer.wer(ref, hyp) for ref, hyp in zip(valid["ref_norm"], valid["hyp_norm"])
    ]

    overall_wer = jiwer.wer(valid["ref_norm"].tolist(), valid["hyp_norm"].tolist())
    return overall_wer, valid


def main() -> None:
    cfg = load_config()
    results_dir = PROJECT_ROOT / cfg["output"]["results_dir"]
    normalizer = build_normalizer(cfg["model"]["name"])

    summary_rows = []
    for run in cfg["runs"]:
        name = run["name"]
        pred_path = results_dir / f"predictions-{name}.csv"
        if not pred_path.exists():
            print(f"[skip] {pred_path.name} not found (run not executed yet)")
            continue

        predictions = pd.read_csv(pred_path)
        overall_wer, detailed = compute_wer(predictions, normalizer)
        detailed.to_csv(results_dir / f"wer-{name}.csv", index=False)

        summary_rows.append({
            "run": name,
            "n_samples": len(detailed),
            "overall_wer_micro": overall_wer,
            "mean_wer_macro": detailed["wer"].mean(),
        })
        print(f"[done] {name}: micro WER = {overall_wer*100:.2f}%   (n={len(detailed)})")

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(results_dir / "wer-summary.csv", index=False)
        print()
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
