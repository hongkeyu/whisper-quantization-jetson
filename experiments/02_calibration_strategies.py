"""实验 2: 用 50 条 LibriSpeech 音频抓 fc2 输入激活, 对比 MinMax/Percentile/KL 三种 calibration 策略。"""

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

from utils import InputCapture, get_module, load_whisper_fp32

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent / "figures"

MODEL_NAME = "openai/whisper-small"          # 实验 2 用 small (forward 速度优先)
TARGET_LAYER = "model.encoder.layers.6.fc2"  # GELU 之后, fc2 的 *输入*
NUM_SAMPLES = 50


def collect_activations(model, processor, manifest, target_name: str, num_samples: int) -> torch.Tensor:
    """跑 forward, 用 hook 抓 target_name 的输入激活, 返回 shape (total_tokens, hidden_dim)。"""
    capture = InputCapture()
    target = get_module(model, target_name)
    hook = target.register_forward_pre_hook(capture, with_kwargs=True)

    try:
        for row in tqdm(manifest.head(num_samples).itertuples(index=False), total=num_samples, desc="forward"):
            audio, sr = sf.read(PROJECT_ROOT / row.audio_path)
            features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
            with torch.no_grad():
                model.model.encoder(features)
    finally:
        hook.remove()

    return capture.stack()


# ===== Calibration 策略 =====

def calibrate_minmax(x: torch.Tensor) -> float:
    """策略 A: 直接取 |x|.max 当 threshold。简单但被 outlier 主导。"""
    return x.abs().max().item()


def calibrate_percentile(x: torch.Tensor, pct: float = 99.9) -> float:
    """策略 B: 取 |x| 的指定分位数, 故意截断 outlier。"""
    return float(np.percentile(x.abs().flatten().numpy(), pct))


def smoothquant_factors(X: torch.Tensor, W: torch.Tensor, alpha: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    """计算 SmoothQuant 每个 input channel 的迁移因子 s_k。
    X: (tokens, in_features)
    W: (out_features, in_features)
    返回 s: (in_features,) — Y 数学不变, 但 X/s 的 outlier 被抹平、W*s 的对应 channel 增大。
    """
    x_max = X.abs().amax(dim=0).clamp(min=eps)   # (in_features,)
    w_max = W.abs().amax(dim=0).clamp(min=eps)   # (in_features,)
    s = (x_max ** alpha) / (w_max ** (1 - alpha))
    return s.clamp(min=eps)


def calibrate_kl(x: torch.Tensor, num_bins: int = 2048, num_quant_bins: int = 128, eps: float = 1e-10) -> float:
    """策略 C: TensorRT 风格的 KL divergence calibration。找 KL(P' || Q) 最小的 threshold。"""
    abs_x = x.abs().flatten().numpy().astype(np.float64)
    max_val = abs_x.max()

    # 步骤 1: 精细直方图 P
    H, edges = np.histogram(abs_x, bins=num_bins, range=(0.0, max_val))
    H = H.astype(np.float64)

    best_kl = np.inf
    best_threshold = max_val

    for i in range(num_quant_bins, num_bins + 1):
        # 步骤 3a: 截断的参考分布 P'
        P = H[:i].copy()
        if i < num_bins:
            P[-1] += H[i:].sum()  # 把溢出的部分加到最后一个 bin (saturation)

        # 步骤 3b: 把 P' 压缩到 num_quant_bins, 再展开回 i 个 bin
        # 把 i 个 bin 平均切成 num_quant_bins 组, 每组合并
        # (用 np.array_split 处理 i 不能整除 num_quant_bins 的情况)
        groups = np.array_split(P, num_quant_bins)
        Q = np.zeros(i)
        idx = 0
        for g in groups:
            n = len(g)
            nz = (g > 0).sum()
            if nz > 0:
                # 把这一组的总质量, 平均铺到 P 非零的位置 (TRT 的 trick)
                fill = g.sum() / nz
                Q[idx:idx+n] = np.where(g > 0, fill, 0.0)
            idx += n

        # 步骤 3c: 算 KL(P' || Q) — 加 eps 防 log(0)
        P_smooth = P + eps
        Q_smooth = Q + eps
        P_norm = P_smooth / P_smooth.sum()
        Q_norm = Q_smooth / Q_smooth.sum()
        kl = float(np.sum(P_norm * np.log(P_norm / Q_norm)))

        if kl < best_kl:
            best_kl = kl
            best_threshold = float(edges[i])

    return best_threshold


# ===== 公共量化器: 用给定 threshold 做对称 INT8 量化 =====

def quantize_with_threshold(x: torch.Tensor, threshold: float, num_bits: int = 8) -> tuple[torch.Tensor, float]:
    """给定 threshold (= clip range 的 |max|), 做对称 INT8 量化, 返回 (x_hat, scale)。"""
    qmax = 2**(num_bits - 1) - 1   # 127
    qmin = -qmax - 1               # -128
    scale = threshold / qmax
    q = torch.round(x / scale).clamp(qmin, qmax)
    x_hat = q * scale
    return x_hat, scale


def stats(x: torch.Tensor, x_hat: torch.Tensor, threshold: float) -> dict:
    err = (x - x_hat).flatten()
    saturated = (x.abs() > threshold).sum().item()
    return {
        "mse": (err**2).mean().item(),
        "max_abs_err": err.abs().max().item(),
        "snr_db": (10 * torch.log10(x.var() / err.var())).item(),
        "saturated": saturated,
        "saturated_pct": 100.0 * saturated / x.numel(),
    }


def plot_activation_distribution(X: torch.Tensor, save_path: Path) -> None:
    x_np = X.flatten().numpy()
    abs_x = np.abs(x_np)

    _, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 全范围线性直方图
    axes[0, 0].hist(x_np, bins=300, color="#3b7dd8", alpha=0.85)
    axes[0, 0].set_title(f"Full range (linear)  —  min={x_np.min():.2f}, max={x_np.max():.2f}")
    axes[0, 0].set_xlabel("activation value")
    axes[0, 0].set_ylabel("count")

    # 全范围 log-y, 看长尾
    axes[0, 1].hist(x_np, bins=300, color="#d8773b", alpha=0.85)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Full range (log y)  —  outlier tail visible")
    axes[0, 1].set_xlabel("activation value")
    axes[0, 1].set_ylabel("count (log)")

    # 放大 [-1, 1] 看主体
    bulk = x_np[(x_np > -1) & (x_np < 1)]
    axes[1, 0].hist(bulk, bins=300, color="#3bd87d", alpha=0.85)
    axes[1, 0].set_title(f"Zoom to [-1, 1]  —  contains {100*len(bulk)/len(x_np):.2f}% of values")
    axes[1, 0].set_xlabel("activation value")
    axes[1, 0].set_ylabel("count")

    # |x| 的 ECDF (经验累积分布), 看每个 |x| 阈值能覆盖多少数据
    sorted_abs = np.sort(abs_x)
    cdf = np.linspace(0, 1, len(sorted_abs))
    axes[1, 1].plot(sorted_abs, cdf, color="#9333ea")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("|x| threshold (log)")
    axes[1, 1].set_ylabel("fraction of data covered")
    axes[1, 1].set_title("|x| ECDF — what threshold covers what fraction")
    axes[1, 1].axhline(0.99, color="red", linestyle="--", alpha=0.5, label="99%")
    axes[1, 1].axhline(0.999, color="orange", linestyle="--", alpha=0.5, label="99.9%")
    axes[1, 1].axhline(0.9999, color="green", linestyle="--", alpha=0.5, label="99.99%")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_calibration_comparison(X: torch.Tensor, results: list[dict], save_path: Path) -> None:
    abs_x = X.abs().flatten().numpy()
    base = {"MinMax": "#3b7dd8", "P99.9": "#d8773b", "KL": "#3bd87d"}
    colors = {**base, **{k + "+SQ": v for k, v in base.items()}}

    _, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # 左: |x| 分布 (log y) + 三条 threshold 线
    axes[0].hist(abs_x, bins=300, color="#999", alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("|x|")
    axes[0].set_ylabel("count (log)")
    axes[0].set_title("Activation |x| distribution with chosen thresholds")
    for r in results:
        axes[0].axvline(r["T"], color=colors[r["name"]], linestyle="--", linewidth=2,
                        label=f"{r['name']} T={r['T']:.2f}  (X.SNR={r['x_snr']:.1f}  Y.SNR={r['y_snr']:.1f} dB)")
    axes[0].legend(fontsize=9)

    # 右: 输入 vs 输出 SNR 对比 (并列柱)
    names = [r["name"] for r in results]
    x_snrs = [r["x_snr"] for r in results]
    y_snrs = [r["y_snr"] for r in results]
    import numpy as np
    x_pos = np.arange(len(names))
    w = 0.38
    axes[1].bar(x_pos - w/2, x_snrs, w, label="X SNR (input MSE)", color="#999")
    axes[1].bar(x_pos + w/2, y_snrs, w, label="Y SNR (output MSE)", color="#9333ea")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("SNR (dB)")
    axes[1].set_title("Input vs output SNR — output is the real metric")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    for i, (xs, ys) in enumerate(zip(x_snrs, y_snrs)):
        axes[1].text(i - w/2, xs, f"{xs:.1f}", ha="center", va="bottom", fontsize=8)
        axes[1].text(i + w/2, ys, f"{ys:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def save_activation_stats(X: torch.Tensor, save_path: Path) -> None:
    x_np = X.flatten().numpy()
    abs_x_np = np.abs(x_np)

    lines = [
        f"=== activation tensor at {TARGET_LAYER} (input) ===",
        f"model:        {MODEL_NAME}",
        f"# audio:      {NUM_SAMPLES}",
        f"shape:        {tuple(X.shape)}  (total_tokens, hidden_dim)",
        f"total elems:  {X.numel():,}",
        f"",
        f"--- value range ---",
        f"min:          {x_np.min():+.6f}",
        f"max:          {x_np.max():+.6f}",
        f"mean:         {x_np.mean():+.6f}",
        f"std:          {x_np.std():.6f}",
        f"|x|.max:      {abs_x_np.max():.6f}",
        f"max/std:      {abs_x_np.max() / x_np.std():.1f}  (>10 表示长尾)",
        f"",
        f"--- |x| percentiles ---",
    ]
    for p in [50, 75, 90, 95, 99, 99.5, 99.9, 99.95, 99.99, 99.999, 100]:
        v = np.percentile(abs_x_np, p)
        lines.append(f"  p{p:>7}:  {v:.4f}")

    save_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    print(f"loading {MODEL_NAME} ...")
    model = load_whisper_fp32(MODEL_NAME)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)

    manifest = pd.read_csv(PROJECT_ROOT / "data/librispeech_subset/manifest.csv")
    print(f"collecting activations at {TARGET_LAYER} from {NUM_SAMPLES} audio samples ...")
    X = collect_activations(model, processor, manifest, TARGET_LAYER, NUM_SAMPLES)

    print(f"\n=== activation tensor at fc2 input ===")
    print(f"shape:       {tuple(X.shape)}  (total_tokens, hidden_dim)")
    print(f"total elems: {X.numel():,}")
    print(f"dtype:       {X.dtype}")
    print(f"min:         {X.min().item():+.4f}")
    print(f"max:         {X.max().item():+.4f}")
    print(f"mean:        {X.mean().item():+.4f}")
    print(f"std:         {X.std().item():.4f}")
    print(f"|x|.max:     {X.abs().max().item():.4f}")

    # 长尾感知: 看几个分位数 (用 numpy 处理超大 tensor)
    abs_x_np = X.abs().flatten().numpy()
    pcts = [50, 90, 99, 99.9, 99.99, 100]
    print(f"\n|x| percentiles:")
    for p in pcts:
        v = np.percentile(abs_x_np, p)
        print(f"  p{p:>5}:  {v:.4f}")

    # 存档供慢慢看
    FIG_DIR.mkdir(exist_ok=True)
    plot_activation_distribution(X, FIG_DIR / "02_activation_distribution.png")
    save_activation_stats(X, FIG_DIR / "02_activation_stats.txt")
    print(f"\n[saved] {FIG_DIR / '02_activation_distribution.png'}")
    print(f"[saved] {FIG_DIR / '02_activation_stats.txt'}")

    # === 三种 calibration 策略对比 ===
    print("\n=== calibrating ...===")
    T_minmax = calibrate_minmax(X)
    print(f"  MinMax     threshold = {T_minmax:.4f}")
    T_pct = calibrate_percentile(X, pct=99.9)
    print(f"  P99.9      threshold = {T_pct:.4f}")
    T_kl = calibrate_kl(X)
    print(f"  KL (TRT)   threshold = {T_kl:.4f}")

    # 拿到 fc2 的权重, 算 baseline output Y = X @ W^T
    W = get_module(model, TARGET_LAYER).weight.detach()  # (768, 3072)
    Y_fp32 = X @ W.T                                      # (total_tokens, 768)

    # === SmoothQuant: 计算迁移因子, 看 X' 的分布变化 ===
    s = smoothquant_factors(X, W, alpha=0.5)
    X_smooth = X / s                     # broadcast: (75000, 3072) / (3072,)
    W_smooth = W * s                     # broadcast: (768, 3072) * (3072,)

    print(f"\n=== SmoothQuant (alpha=0.5) ===")
    print(f"original X |max| = {X.abs().max():.3f},   smoothed X' |max| = {X_smooth.abs().max():.3f}  "
          f"(× {X.abs().max() / X_smooth.abs().max():.1f} smaller)")
    print(f"original W |max| = {W.abs().max():.4f}, smoothed W' |max| = {W_smooth.abs().max():.4f}  "
          f"(× {W_smooth.abs().max() / W.abs().max():.1f} larger)")
    print(f"sanity check: ||(X @ W^T) - (X' @ W'^T)|| = {(X @ W.T - X_smooth @ W_smooth.T).abs().max():.2e}  (应该 ~0)")

    print("\n=== quantize-dequantize, compare 输入 MSE  vs  输出 MSE  ===")
    results = []
    # 原始 (无 SmoothQuant)
    for name, T_fn in [("MinMax", lambda x: calibrate_minmax(x)),
                       ("P99.9",  lambda x: calibrate_percentile(x, 99.9)),
                       ("KL",     lambda x: calibrate_kl(x))]:
        T = T_fn(X)
        X_hat, scale = quantize_with_threshold(X, T, num_bits=8)
        Y_hat = X_hat @ W.T
        s_x = stats(X, X_hat, T)
        out_err = (Y_fp32 - Y_hat).flatten()
        out_snr = (10 * torch.log10(Y_fp32.var() / out_err.var())).item()
        results.append({"name": name, "smoothed": False, "T": T, "scale": scale,
                        "x_snr": s_x["snr_db"], "x_mse": s_x["mse"], "sat_pct": s_x["saturated_pct"],
                        "y_snr": out_snr, "y_mse": (out_err**2).mean().item()})

    # SmoothQuant 之后 (X' 上做 calibration, Y' 用 W' 算)
    for name, T_fn in [("MinMax", lambda x: calibrate_minmax(x)),
                       ("P99.9",  lambda x: calibrate_percentile(x, 99.9)),
                       ("KL",     lambda x: calibrate_kl(x))]:
        T = T_fn(X_smooth)
        Xs_hat, scale = quantize_with_threshold(X_smooth, T, num_bits=8)
        Y_hat = Xs_hat @ W_smooth.T
        s_x = stats(X_smooth, Xs_hat, T)
        out_err = (Y_fp32 - Y_hat).flatten()
        out_snr = (10 * torch.log10(Y_fp32.var() / out_err.var())).item()
        results.append({"name": name + "+SQ", "smoothed": True, "T": T, "scale": scale,
                        "x_snr": s_x["snr_db"], "x_mse": s_x["mse"], "sat_pct": s_x["saturated_pct"],
                        "y_snr": out_snr, "y_mse": (out_err**2).mean().item()})

    print(f"{'strategy':<12} {'T':>9} {'X.SNR(dB)':>11} {'X.MSE':>11} {'sat %':>7}  |  {'Y.SNR(dB)':>11} {'Y.MSE':>11}")
    print("-" * 94)
    for r in results:
        print(f"{r['name']:<12} {r['T']:>9.3f} {r['x_snr']:>11.2f} {r['x_mse']:>11.3e} {r['sat_pct']:>6.3f}%  |  {r['y_snr']:>11.2f} {r['y_mse']:>11.3e}")

    plot_calibration_comparison(X, results, FIG_DIR / "02_calibration_comparison.png")
    print(f"\n[saved] {FIG_DIR / '02_calibration_comparison.png'}")


if __name__ == "__main__":
    main()
