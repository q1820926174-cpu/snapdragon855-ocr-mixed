# Snapdragon 855 mixed-precision OCR

**English** | [简体中文](README.zh-CN.md)

Run PP-OCRv6 text recognition across the **HTA/NPU and Adreno GPU** on a Samsung Galaxy S10 SCV41 running native Linux. This project shares the integration code, model weights and build recipes used in the experiments.

**Status: experimental. The existing pure-GPU setup remains the best everyday choice on the tested phone.** The mixed profiles expose the accuracy, throughput and energy trade-offs for further development.

[Download models](https://github.com/q1820926174-cpu/snapdragon855-ocr-mixed/releases/tag/v0.1.0) · [Build guide](docs/build.md) · [Benchmark details](docs/benchmarks.md) · [Model card](docs/model-card.md)

## What the measurements show

These results cover **310 cropped text images** at a fixed 48×640 input size. They measure recognition after model initialization.

| Configuration | Correct / 310 | Crops per second ↑ | Joules per crop ↓ |
|---|---:|---:|---:|
| CPU baseline | 255 | 2.887 | 1.750 |
| **Existing pure GPU** | **256** | **4.584** | **0.991** |
| Mixed `quality` | 255 | 2.428 | 1.618 |
| Mixed `fast` | 252 | 5.305 | 0.736 |

`quality` retained all samples the CPU recognized correctly, but ran slower and used more energy per crop than pure GPU. `fast` improved throughput and energy use, while losing five CPU-correct samples. Its total correct count fell by three because it also gained two other samples.

Energy is a **whole-system rail estimate**, including the desktop. Baseline temperatures differed. The roughly 7.5% energy reduction of `quality` versus CPU needs controlled repetition; the roughly 58% reduction of `fast` comes with accuracy regressions. See the [measurement method and limits](docs/benchmarks.md).

The corpus is a development regression set used during engineering choices. These results do not establish lossless quantization on unseen images.

## What you need

| Component | Tested setup |
|---|---|
| Phone | Samsung SCV41, Snapdragon 855 / SM8150, Adreno 640 |
| Phone OS | Native Debian 12 with working NPU, ION, FastRPC and KGSL Mesa |
| External software | Your authorized SNPE 1.59.0.3230 SDK/runtime and compatible device firmware |
| Preparation host | x86-64 Linux or WSL; Python 3.11 for preprocessing and asset checks; compatible Python 3.6 for the SDK |

The release provides integration source and open model tensors. Obtain the Qualcomm SDK/runtime and firmware separately, then generate the HTA containers locally. Only SCV41 has been validated. Device flashing and platform-driver installation are outside this project.

## Get started

### 1. Download and verify the models

On the preparation host, with Git and GitHub CLI installed:

```bash
git clone https://github.com/q1820926174-cpu/snapdragon855-ocr-mixed.git
cd snapdragon855-ocr-mixed

gh release download v0.1.0 -R q1820926174-cpu/snapdragon855-ocr-mixed \
  -p sm8150-open-models-v0.1.0.tar.gz -p SHA256SUMS
sha256sum -c SHA256SUMS
tar -xzf sm8150-open-models-v0.1.0.tar.gz

python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_models.py models
```

The model asset contains partitioned ONNX graphs, a frozen W8 weight grid, quantization parameters, GPU ncnn models, affine/sparse corrections, the dictionary and a reference FP32 model. The [model card](docs/model-card.md) explains how these pieces fit together.

### 2. Build the containers and runtime

Follow the [build guide](docs/build.md) to:

1. Generate and audit the SM8150 HTA containers on the x86-64 host using your SDK.
2. Build the aarch64 executors with the pinned ncnn version and supplied patch.
3. Check the phone's platform prerequisites and transfer the prepared files.

### 3. Prepare text crops and run on the phone

On the host, prepare images that have already been cropped to text regions:

```bash
python scripts/prepare_inputs.py /path/to/crops /path/to/raw \
  --device-prefix /path/on/phone/raw
```

Copy the generated raw files and `input.tsv` to the phone. From the prepared repository directory **on the phone**, run:

```bash
sudo bash run.sh quality /path/on/phone/raw/input.tsv /path/to/result.jsonl 310

# Optional: faster profile with known accuracy regressions.
sudo bash run.sh fast /path/on/phone/raw/input.tsv /path/to/result-fast.jsonl 310
```

This is text recognition for cropped regions. Add your own text detection, orientation handling and page analysis as needed. Each invocation accepts 1–310 crops. The runtime uses an exclusive NPU lock and a 170-second timeout, and restores the previous NPU firmware power state on exit.

## How the profiles work

| Profile | NPU work | GPU work |
|---|---|---|
| `quality` | 12 large W8A8 convolutions; two passes per layer refine the input quantization residual | Remaining operators and storage use FP32; sparse outlier correction |
| `fast` | 16 large W8A8 convolutions; one pass per layer | FP16 packed CNN storage with FP32 arithmetic; FP32 classifier; fused data conversion |

The selected convolutions account for about 54.9% and 75.5% of the original convolution MACs, respectively. The process shares 12 or 16 NPU contexts across four workers and bounds in-flight requests at ten. CPU fallback is disabled. **Two INT8 passes do not mean native W8A16 hardware support.**

## Documentation and validation

| Read about | English | 简体中文 |
|---|---|---|
| Build and platform setup | [Build guide](docs/build.md) | [构建指南](docs/build.zh-CN.md) |
| Results and energy interpretation | [Benchmarks](docs/benchmarks.md) | [测试与能耗](docs/benchmarks.zh-CN.md) |
| Model design and evaluation limits | [Model card](docs/model-card.md) | [模型说明](docs/model-card.zh-CN.md) |
| Licenses and release contents | [Licensing](docs/licensing.md) | [许可与发布范围](docs/licensing.zh-CN.md) |

Both original profiles and their reconstruction from the public source and weights were replayed on all 310 crops with identical text outputs. See [recheck results](benchmarks/recheck.json) and [benchmark data](benchmarks/results.json). CI checks source and input contracts; hardware validation uses the real phone.

Original integration code and PaddleOCR-derived model assets use Apache-2.0; the ncnn patch retains BSD-3-Clause attribution. See [NOTICE](NOTICE) and [licenses/](licenses/). Qualcomm dependencies remain under their separate licenses.
