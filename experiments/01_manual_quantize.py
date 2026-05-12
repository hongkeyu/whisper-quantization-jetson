"""实验 1: 从零实现量化/反量化, 理解 scale 和 zero_point。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import torch

from utils import list_linear_layers, load_whisper_fp32

FIG_DIR = Path(__file__).resolve().parent / "figures"

TARGET_LAYER = "model.encoder.layers.16.fc1"  # whisper-large: 32 层中间, FFN (1280 -> 5120)


def asymmetric_quantize(x: torch.Tensor, num_bits: int = 8) -> tuple[torch.Tensor, float, int]:
    """非对称量化: FP32 -> uint{num_bits}. 返回 (q, scale, zero_point)。"""
    qmin, qmax = 0, 2**num_bits - 1
    x_min, x_max = x.min().item(), x.max().item()
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = round(qmin - x_min / scale)
    zero_point = max(qmin, min(qmax, zero_point))  # clamp to valid range

    q = torch.round(x / scale + zero_point).clamp(qmin, qmax)
    return q, scale, zero_point


def asymmetric_dequantize(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """反量化: uint{num_bits} -> FP32 重建。"""
    return (q - zero_point) * scale


def symmetric_quantize(x: torch.Tensor, num_bits: int = 8) -> tuple[torch.Tensor, float]:
    """对称量化: FP32 -> int{num_bits}, zero_point 固定为 0。"""
    qmax = 2**(num_bits - 1) - 1   # int8: 127
    qmin = -qmax - 1               # int8: -128
    abs_max = x.abs().max().item()
    scale = abs_max / qmax         # 用 |x|.max 决定范围, 对称
    q = torch.round(x / scale).clamp(qmin, qmax)
    return q, scale


def symmetric_dequantize(q: torch.Tensor, scale: float) -> torch.Tensor:
    return q * scale


def symmetric_quantize_per_channel(x: torch.Tensor, num_bits: int = 8, channel_dim: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """对称 per-channel 量化: 每个 output channel 一个独立 scale。"""
    qmax = 2**(num_bits - 1) - 1
    qmin = -qmax - 1

    # 沿 channel_dim 之外的所有维度求 |max|, 保留 channel_dim 用于 broadcast
    other_dims = [d for d in range(x.dim()) if d != channel_dim]
    abs_max = x.abs().amax(dim=other_dims, keepdim=True)  # shape (5120, 1) for (5120, 1280)
    scales = abs_max / qmax                                # shape (5120, 1), 每行一个 scale

    q = torch.round(x / scales).clamp(qmin, qmax)
    return q, scales


def symmetric_dequantize_per_channel(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return q * scales


def reconstruction_stats(x: torch.Tensor, x_hat: torch.Tensor) -> dict:
    err = (x - x_hat).flatten()
    return {
        "mse": (err**2).mean().item(),
        "max_abs_err": err.abs().max().item(),
        "snr_db": 10 * torch.log10(x.var() / err.var()).item(),
    }


def plot_quantize_demo(x: torch.Tensor, x_hat: torch.Tensor, q: torch.Tensor, save_path: Path) -> None:
    _, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    axes[0].hist(x.flatten().numpy(), bins=200, color="#3b7dd8", alpha=0.85)
    axes[0].set_title(f"Original FP32 weights ({x.numel():,} values)")
    axes[0].set_xlabel("weight value")
    axes[0].set_ylabel("count")

    axes[1].hist(q.flatten().numpy(), bins=256, color="#d8773b", alpha=0.85)
    axes[1].set_title("After quantization (uint8 integers)")
    axes[1].set_xlabel("integer code [0, 255]")
    axes[1].set_xlim(0, 255)

    err = (x - x_hat).flatten().numpy()
    axes[2].hist(err, bins=200, color="#3bd87d", alpha=0.85)
    axes[2].set_title(f"Reconstruction error  (max |err| = {abs(err).max():.5f})")
    axes[2].set_xlabel("x − x̂")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_bit_sweep(rows: list[dict], save_path: Path) -> None:
    bits = [r["bits"] for r in rows]
    pt = [r["per_tensor"] for r in rows]
    pc = [r["per_channel"] for r in rows]

    # 6 dB/bit 理论参考线 (锚点: per-tensor INT8 实测值)
    anchor_b, anchor_snr = 8, [r for r in rows if r["bits"] == 8][0]["per_tensor"]
    ideal = [anchor_snr + 6.02 * (b - anchor_b) for b in bits]

    _, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(bits, pt, "o-", label="per-tensor (symmetric)", color="#3b7dd8")
    ax.plot(bits, pc, "s-", label="per-channel (symmetric)", color="#d8773b")
    ax.plot(bits, ideal, "--", label="6 dB/bit slope (anchored at INT8 per-tensor)", color="gray", alpha=0.7)
    ax.set_xlabel("bit width")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("Whisper-large fc1.weight — quantization SNR vs bit width")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def main() -> None:
    model = load_whisper_fp32()
    layers = dict(list_linear_layers(model))

    print(f"encoder Linear layer count: {len(layers)}")
    print(f"\n=== {TARGET_LAYER} ===")
    W = layers[TARGET_LAYER].weight.detach().clone()
    print(f"shape:    {tuple(W.shape)}")
    print(f"params:   {W.numel():,}")
    print(f"dtype:    {W.dtype}")
    print(f"min:      {W.min().item():+.6f}")
    print(f"max:      {W.max().item():+.6f}")
    print(f"mean:     {W.mean().item():+.6f}")
    print(f"std:      {W.std().item():.6f}")
    print(f"|w|.max:  {W.abs().max().item():.6f}")

    # Asymmetric uint8
    q_a, scale_a, zp_a = asymmetric_quantize(W, num_bits=8)
    W_hat_a = asymmetric_dequantize(q_a, scale_a, zp_a)
    stats_a = reconstruction_stats(W, W_hat_a)

    # Symmetric int8 (per-tensor)
    q_s, scale_s = symmetric_quantize(W, num_bits=8)
    W_hat_s = symmetric_dequantize(q_s, scale_s)
    stats_s = reconstruction_stats(W, W_hat_s)

    # Symmetric int8 (per-channel)
    q_pc, scales_pc = symmetric_quantize_per_channel(W, num_bits=8, channel_dim=0)
    W_hat_pc = symmetric_dequantize_per_channel(q_pc, scales_pc)
    stats_pc = reconstruction_stats(W, W_hat_pc)

    print("\n=== INT8 quantization comparison (whisper-large fc1, 5120 x 1280) ===")
    print(f"{'metric':<14} {'asym/per-T':>14} {'sym/per-T':>14} {'sym/per-CH':>14}")
    print(f"{'unique q':<14} {len(torch.unique(q_a)):>14} {len(torch.unique(q_s)):>14} {len(torch.unique(q_pc)):>14}")
    print(f"{'MSE':<14} {stats_a['mse']:>14.3e} {stats_s['mse']:>14.3e} {stats_pc['mse']:>14.3e}")
    print(f"{'max |err|':<14} {stats_a['max_abs_err']:>14.3e} {stats_s['max_abs_err']:>14.3e} {stats_pc['max_abs_err']:>14.3e}")
    print(f"{'SNR (dB)':<14} {stats_a['snr_db']:>14.2f} {stats_s['snr_db']:>14.2f} {stats_pc['snr_db']:>14.2f}")
    print(f"\nper-channel scales:  count={scales_pc.numel()}  range=[{scales_pc.min().item():.6f}, {scales_pc.max().item():.6f}]")

    # Bit-width sweep: 验证 +1 bit ≈ +6 dB
    bits = [16, 12, 10, 8, 6, 5, 4, 3, 2]
    rows = []
    for b in bits:
        q_t, scale_t = symmetric_quantize(W, num_bits=b)
        snr_t = reconstruction_stats(W, symmetric_dequantize(q_t, scale_t))["snr_db"]
        q_c, scales_c = symmetric_quantize_per_channel(W, num_bits=b, channel_dim=0)
        snr_c = reconstruction_stats(W, symmetric_dequantize_per_channel(q_c, scales_c))["snr_db"]
        rows.append({"bits": b, "per_tensor": snr_t, "per_channel": snr_c})

    print(f"\n=== bit-width sweep (symmetric) ===")
    print(f"{'bits':>4} {'per-tensor':>12} {'per-channel':>13} {'gap':>7}")
    for r in rows:
        print(f"{r['bits']:>4} {r['per_tensor']:>12.2f} {r['per_channel']:>13.2f} {r['per_channel']-r['per_tensor']:>+7.2f}")

    plot_bit_sweep(rows, FIG_DIR / "01_bit_width_sweep.png")
    print(f"[saved] {FIG_DIR / '01_bit_width_sweep.png'}")

    FIG_DIR.mkdir(exist_ok=True)
    plot_quantize_demo(W, W_hat_a, q_a, FIG_DIR / "01_asymmetric_int8.png")
    print(f"\n[saved] {FIG_DIR / '01_asymmetric_int8.png'}")


if __name__ == "__main__":
    main()
