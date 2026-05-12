"""实验 3: per-layer sensitivity analysis - 一次量化一层, 测对 encoder 输出的影响。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import WhisperProcessor

from utils import get_module, list_linear_layers, load_whisper_fp32

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent / "figures"

MODEL_NAME = "openai/whisper-small"
NUM_AUDIO = 10              # 用 10 条音频跑敏感度 (vs 50 条精度但更快)
NUM_BITS = 8                # 每个 Linear 都按 INT8 per-channel symmetric 量化


def symmetric_quantize_per_channel(W: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """Per-output-channel 对称量化 + 反量化, 返回还原后的 W_hat (实验 1 同款)。"""
    qmax = 2**(num_bits - 1) - 1
    abs_max = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scales = abs_max / qmax
    q = torch.round(W / scales).clamp(-qmax - 1, qmax)
    return q * scales


def encoder_outputs(model, mel_features_list: list[torch.Tensor]) -> torch.Tensor:
    """对每条预计算好的 mel features 跑 encoder, 拼接所有输出。"""
    outs = []
    for feats in mel_features_list:
        with torch.no_grad():
            out = model.model.encoder(feats).last_hidden_state  # (1, 1500, hidden)
        outs.append(out.cpu())
    return torch.cat(outs, dim=1)  # (1, total_tokens, hidden)


def main() -> None:
    print(f"loading {MODEL_NAME} ...")
    model = load_whisper_fp32(MODEL_NAME)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    manifest = pd.read_csv(PROJECT_ROOT / "data/librispeech_subset/manifest.csv")

    # 预计算 mel features (一次), 避免每次循环都跑 librosa
    print(f"preprocessing {NUM_AUDIO} audio samples into mel features...")
    mels = []
    for row in manifest.head(NUM_AUDIO).itertuples(index=False):
        audio, sr = sf.read(PROJECT_ROOT / row.audio_path)
        feats = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
        mels.append(feats)

    # 1. Baseline: 所有层都是 FP32
    print("computing baseline (FP32) encoder outputs...")
    Y_baseline = encoder_outputs(model, mels)
    print(f"  baseline output shape: {tuple(Y_baseline.shape)}")
    base_var = Y_baseline.var().item()

    # 2. 对每个 Linear 层, 量化-测量-还原
    layers = list_linear_layers(model, prefix="model.encoder")
    print(f"\nrunning sensitivity analysis on {len(layers)} encoder Linear layers...")
    results = []
    for name, mod in tqdm(layers, desc="layers"):
        original = mod.weight.data.clone()
        mod.weight.data = symmetric_quantize_per_channel(mod.weight.data, NUM_BITS)

        Y_q = encoder_outputs(model, mels)
        err = (Y_baseline - Y_q).flatten()
        mse = (err ** 2).mean().item()
        err_var = err.var().item()
        snr_db = 10 * np.log10(base_var / err_var) if err_var > 0 else float("inf")

        results.append({
            "layer": name,
            "layer_idx": int(name.split(".layers.")[1].split(".")[0]),
            "kind": name.split(".")[-1] if "." in name else name,
            "mse": mse,
            "snr_db": snr_db,
        })

        mod.weight.data = original  # 还原

    df = pd.DataFrame(results).sort_values("mse", ascending=False).reset_index(drop=True)

    # 3. 输出 + 图
    FIG_DIR.mkdir(exist_ok=True)
    df.to_csv(FIG_DIR / "03_sensitivity.csv", index=False)

    print(f"\n=== top 10 most sensitive layers ===")
    print(df.head(10)[["layer", "mse", "snr_db"]].to_string(index=False))
    print(f"\n=== bottom 10 least sensitive layers ===")
    print(df.tail(10)[["layer", "mse", "snr_db"]].to_string(index=False))

    plot_sensitivity(df, FIG_DIR / "03_sensitivity.png")
    print(f"\n[saved] {FIG_DIR / '03_sensitivity.csv'}")
    print(f"[saved] {FIG_DIR / '03_sensitivity.png'}")


def plot_sensitivity(df: pd.DataFrame, save_path: Path) -> None:
    # 颜色映射: 不同 layer kind
    kind_colors = {
        "k_proj": "#1f77b4", "v_proj": "#ff7f0e", "q_proj": "#2ca02c", "out_proj": "#d62728",
        "fc1": "#9467bd", "fc2": "#8c564b",
    }

    df_sorted = df.sort_values(["layer_idx", "kind"]).reset_index(drop=True)

    _, axes = plt.subplots(2, 1, figsize=(14, 9))

    # 上图: 全部 72 层的 MSE (按 layer 顺序), 颜色 = kind
    ax = axes[0]
    for kind, color in kind_colors.items():
        sub = df_sorted[df_sorted["kind"] == kind]
        ax.scatter(sub["layer_idx"], sub["mse"], label=kind, color=color, s=70, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("encoder layer index (0-11)")
    ax.set_ylabel("output MSE  (log scale)")
    ax.set_title("Per-layer quantization sensitivity — INT8 per-channel symmetric")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(alpha=0.3)

    # 下图: top-15 + bottom-15 排名条形
    top15 = df.head(15)
    bot15 = df.tail(15).iloc[::-1]
    combined = pd.concat([top15, bot15], ignore_index=True)
    colors = [kind_colors.get(k, "#888") for k in combined["kind"]]
    ax2 = axes[1]
    y_pos = range(len(combined))
    ax2.barh(y_pos, combined["mse"], color=colors)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels([n.replace("model.encoder.", "") for n in combined["layer"]], fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("MSE (log)")
    ax2.set_title("Top-15 most sensitive (top) and bottom-15 least sensitive (bottom)")
    ax2.axhline(14.5, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
