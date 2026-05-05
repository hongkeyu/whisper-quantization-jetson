# Whisper 量化部署对比

在 Jetson Orin Nano 8GB 上对比 Whisper-small 在 FP32 / FP16 / INT8 下的推理速度与识别精度。

## 环境

- 开发: MacBook Pro M1 (CPU 推理，验证流程与精度)
- 部署: Jetson Orin Nano 8GB (CUDA 推理，最终速度数据)
- 备用 GPU: Google Colab (T4/A100，开发期间验证 GPU 路径)

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 流程

1. `src/convert.py` — 下载 Whisper-small 并转换为 CT2 格式 (FP32/FP16/INT8)
2. `src/prepare_data.py` — 准备评测音频 (LibriSpeech 子集)
3. `src/inference.py` — 推理封装
4. `src/evaluate.py` — 计算 WER
5. `src/benchmark.py` — 测量 RTF / 延迟 / 显存
6. `src/compare.py` — 汇总各精度结果

## 配置

所有可调参数集中在 `config.yaml`，跑不同机器只改 `runtime.device` 和 `runtime.compute_type`。
