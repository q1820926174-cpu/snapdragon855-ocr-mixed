# Snapdragon 855 OCR: native GPU and mixed NPU/GPU

**English** | [简体中文](README.zh-CN.md)

Run **PP-OCRv6 text recognition** on a Samsung Galaxy S10 SCV41 with native Linux. The recommended experimental path is now **PNNX → ncnn Vulkan on Adreno 640**. The earlier HTA/NPU + GPU profiles remain available for comparison.

**v0.2:** the native GPU path retains every previously correct regression line, improves throughput by **36.0%** and reduces system energy per crop by **28.3%** versus the previous pure-GPU setup, in the controlled tests below. It needs no Qualcomm SDK or NPU runtime.

[Download v0.2 models](https://github.com/q1820926174-cpu/snapdragon855-ocr-mixed/releases/tag/v0.2.0) · [GPU build/export guide](docs/gpu-native.md) · [Test details](docs/pnnx-benchmarks.md)

## Docker API (v0.3)

A resident GPU HTTP service is now available: **6.250 crops/s with ten concurrent clients** on the tested phone, including upload, decoding and preprocessing. All 310 texts and output hashes match v0.2; models initialized once. The API provides single/batch recognition, bounded queues, optional bearer authentication, metrics and automatic recovery from a worker-process failure.

[Download ARM64 Docker image](https://github.com/q1820926174-cpu/snapdragon855-ocr-mixed/releases/tag/v0.3.0) · [Deployment and API guide](docs/api.md) · [Measured API data](benchmarks/api-v0.3.json)

Models are mounted separately from v0.2. GPU acceleration uses host KGSL Mesa; no SNPE or NPU runtime is bundled. This remains **cropped-text recognition**, not page detection. The source Dockerfile and tested cache/cross-build recipe are included.

## Measured results

Medians of **two 310-crop runs per configuration**, starting after battery temperature was at or below 35.5°C in three checks five seconds apart. The second round reversed order; a common device lock excluded other managed accelerator tests. Initialization is excluded.

| Configuration | Correct / 310 | Crops/s ↑ | System W | J/crop ↓ |
|---|---:|---:|---:|---:|
| Previous pure GPU | 256 | 4.591 | 4.344 | 0.946 |
| **PNNX native GPU** | 257 | 6.243 | 4.233 | 0.678 |
| Mixed `quality` | 255 | 2.463 | 3.697 | 1.502 |
| Mixed `fast` | 252 | 5.243 | 3.727 | 0.711 |

Native GPU retained all 256 old-GPU-correct and all 255 CPU-correct lines. Mixed `quality` preserves CPU-correct lines but is slower; `fast` loses five CPU-correct lines and is not an equal-quality alternative. There were **1 incomplete mixed-runtime attempt(s)**, recorded separately and excluded from performance medians.

Energy is an **ISYS × VSYS whole-system rail estimate**, including the desktop, not isolated GPU power or battery-discharge energy. Two repeats quantify these runs, not statistical certainty across phones. [Raw relative-time power traces and aggregates](benchmarks/pnnx-controlled.json) are provided. Historical CPU results remain in the [earlier report](docs/benchmarks.md); CPU was not rerun in this controlled comparison.

Native GPU energy ranged from 0.641 to 0.716 J/crop. Its range overlaps mixed `fast`; the small median difference does not establish a repeatable energy advantage over that lower-accuracy profile.

These are development regression data, not unseen holdout accuracy. A supplementary 100-crop corpus retained all 87 old-GPU-correct lines, but character errors increased by one; the main corpus improved by one character edit. **FP16 is not claimed to be universally lossless.** See the [precision report](benchmarks/pnnx-precision.json).

## Run the GPU version

On an x86-64 Linux/WSL preparation host, with Git and GitHub CLI:

```bash
git clone https://github.com/q1820926174-cpu/snapdragon855-ocr-mixed.git
cd snapdragon855-ocr-mixed
gh release download v0.2.0 -R q1820926174-cpu/snapdragon855-ocr-mixed \
  -p sm8150-pnnx-gpu-v0.2.0.tar.gz -p SHA256SUMS
sha256sum -c SHA256SUMS
tar -xzf sm8150-pnnx-gpu-v0.2.0.tar.gz
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_gpu_models.py models/gpu-native
```

Build the aarch64 runner with the pinned, patched ncnn dependency using the [GPU guide](docs/gpu-native.md). The phone must already have a working Vulkan driver and GPU access permissions. Only SCV41/native Debian 12 with the tested KGSL Mesa stack has been validated; this repository does not install drivers or flash phones.

Prepare images already cropped to text regions on the host:

```bash
python scripts/prepare_inputs.py /path/to/crops /path/to/raw \
  --device-prefix /path/on/phone/raw
```

Transfer the inputs, repository, GPU models and aarch64 build to the phone. Run **on the phone**:

```bash
bash run-gpu.sh /path/on/phone/raw/input.tsv /path/to/result.jsonl 310
```

This is cropped-text recognition, not full-page OCR. Each batch CLI invocation accepts 1–310 crops. It uses two workers and ten bounded in-flight requests; it does not install a service or change RustDesk or the desktop.

## Why the GPU path changed

The old GPU path reused an expanded SNPE-compatible graph. The new path starts from the original ONNX and uses PNNX 20261020, preserving native GELU, LayerNorm and attention. CNN storage/arithmetic use FP16; the sequence/classifier tail stays FP32. CNN features stay on the GPU, and fused CTC returns only 640 bytes per crop. Model execution has no CPU fallback. The measured improvement combines graph, kernel, precision and transfer changes; it cannot be attributed to PNNX alone.

The original ONNX, graph hashes and standalone exporter are included. Public-source reconstruction was checked against the tested graphs, and the public binary/script replayed all 310 crops. See [recheck data](benchmarks/pnnx-recheck.json).

## Earlier mixed profiles

The separate [v0.1 model asset](https://github.com/q1820926174-cpu/snapdragon855-ocr-mixed/releases/tag/v0.1.0) and [mixed build guide](docs/build.md) remain available. These profiles require your own authorized SNPE 1.59 SDK/runtime and compatible NPU/ION/FastRPC/firmware setup. Build with `-DBUILD_MIXED=ON`, generate HTA containers locally, then run on the qualified phone:

```bash
sudo bash run.sh quality /path/on/phone/raw/input.tsv /path/to/result-quality.jsonl 310
sudo bash run.sh fast /path/on/phone/raw/input.tsv /path/to/result-fast.jsonl 310
```

`quality` uses 12 shared NPU contexts and two INT8 passes per selected convolution, with remaining GPU work in FP32. `fast` uses 16 contexts and one INT8 pass, with FP16 packed CNN storage/FP32 arithmetic. Both use four workers and at most ten crops in flight. Two INT8 passes do not mean native W8A16 hardware support. The original release files are unchanged.

## Documentation

| Read about / 内容 | English | 简体中文 |
|---|---|---|
| Native GPU export/build / 原生 GPU 导出与构建 | [GPU guide](docs/gpu-native.md) | [GPU 指南](docs/gpu-native.zh-CN.md) |
| Controlled tests / 受控测试 | [PNNX benchmarks](docs/pnnx-benchmarks.md) | [PNNX 测试](docs/pnnx-benchmarks.zh-CN.md) |
| Mixed build / 混合版构建 | [Build guide](docs/build.md) | [构建指南](docs/build.zh-CN.md) |
| Original measurements / 早期测量 | [Legacy benchmarks](docs/benchmarks.md) | [早期测试](docs/benchmarks.zh-CN.md) |
| Model and licenses / 模型与许可 | [Model card](docs/model-card.md) · [Licensing](docs/licensing.md) | [模型说明](docs/model-card.zh-CN.md) · [许可](docs/licensing.zh-CN.md) |

Original integration code and PaddleOCR-derived model assets use Apache-2.0; the ncnn patch retains BSD-3-Clause attribution. See [NOTICE](NOTICE) and [licenses/](licenses/). Qualcomm software and firmware, compiled HTA caches, executables, evaluation images/transcriptions and private device-access files are not redistributed.
