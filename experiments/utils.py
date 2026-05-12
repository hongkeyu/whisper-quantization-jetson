"""量化实验共享工具: 加载 PyTorch FP32 Whisper, 抽取真实权重和激活。"""

from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_whisper_fp32(model_name: str = "openai/whisper-large-v3") -> WhisperForConditionalGeneration:
    """加载 HuggingFace 原始 PyTorch FP32 模型 (不是 CT2 压缩版)。"""
    model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    return model


def list_linear_layers(model: torch.nn.Module, prefix: str = "model.encoder") -> list[tuple[str, torch.nn.Linear]]:
    """返回 (name, module) 列表, 只挑指定前缀下的 nn.Linear 层 (排除 embedding/norm)。"""
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.startswith(prefix)
    ]


def get_module(model: torch.nn.Module, dotted_name: str) -> torch.nn.Module:
    """通过 'a.b.c' 形式的层名取出对应 module。"""
    obj = model
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    return obj


class InputCapture:
    """前向 hook: 捕获指定 module 的 *输入* 激活, 多次 forward 累积。"""

    def __init__(self):
        self.batches: list[torch.Tensor] = []

    def __call__(self, _module, args, kwargs):
        # nn.Linear forward(input) -> args == (input,); _module is required by hook signature but unused
        x = args[0] if len(args) > 0 else kwargs.get("input")
        self.batches.append(x.detach().clone().cpu())

    def stack(self) -> torch.Tensor:
        """所有 batch 沿 token 维度拼起来, shape (total_tokens, hidden_dim)。"""
        return torch.cat([b.reshape(-1, b.shape[-1]) for b in self.batches], dim=0)
