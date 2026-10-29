# Native PNNX GPU export and runtime

[Home](../README.md) · **English** | [简体中文](gpu-native.zh-CN.md)

The v0.2 GPU path starts from the original PP-OCRv6 model, before the graph was lowered for the old SNPE converter. It runs entirely through ncnn Vulkan on the tested Adreno 640. **It does not require SNPE, an NPU runtime or root.** The phone must already have a working Vulkan driver and permission to access its GPU.

## Download and verify

Run in a clone of this repository on your preparation host:

```bash
gh release download v0.2.0 -R q1820926174-cpu/snapdragon855-ocr-mixed \
  -p sm8150-pnnx-gpu-v0.2.0.tar.gz -p SHA256SUMS
sha256sum -c SHA256SUMS
tar -xzf sm8150-pnnx-gpu-v0.2.0.tar.gz
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_gpu_models.py models/gpu-native
```

This separate asset includes the original ONNX model, CNN/tail/full ncnn graphs, dictionary, model contract and file hashes. The v0.1 mixed asset remains a separate download. No Qualcomm files, executables, evaluation images or transcriptions are included.

## Reproduce the conversion

The supplied graphs can be used directly. To rebuild them, install the pinned PNNX binary package without its optional Torch frontend dependencies, then run the exporter:

```bash
pip install --no-deps -r requirements-gpu-export.txt
python scripts/export_gpu_pnnx.py \
  --source models/gpu-native/source/original.onnx \
  --output /tmp/ocr-pnnx-export
```

The exporter checks the original source SHA-256, fixes input shape to `[1,3,48,640]`, removes only the terminal Softmax for GPU CTC, and partitions at the original average-pool boundary `[1,768,1,80]`. Retained ONNX initializers are checked byte for byte before PNNX conversion; PNNX may subsequently fold/fuse weights. The original Erf-based GELU is preserved. The full graph has 14 native GELU, five LayerNorm and two MultiHeadAttention operators. A singleton Squeeze unsupported by ncnn Vulkan is replaced by an equivalent fixed-shape Reshape.

An FP32 conversion check against original ONNX Runtime on 20 crops preserved all 80 frame argmaxes per crop; the maximum absolute logit difference was below 0.000153. This checks graph conversion, **not bitwise equivalence of the selected phone FP16 runtime**. See the [precision report](../benchmarks/pnnx-precision.json).

[PNNX documentation](https://github.com/pnnx/pnnx) describes ONNX conversion and standalone binaries. The [legacy onnx2ncnn documentation](https://github.com/Tencent/ncnn/blob/master/tools/onnx/README.md) recommends PNNX for newer conversion workflows.

## Build for the phone

Build the pinned, patched ncnn dependency using [build guide section 2](build.md#2-build-the-aarch64-runtime). For the GPU runner, replace that guide's SNPE-dependent application configure command with:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR="$PWD/_deps/ncnn-install/lib/cmake/ncnn" \
  -DBUILD_MIXED=OFF
cmake --build build -j4
```

For cross compilation add your aarch64 toolchain to both the ncnn and application configure commands. The public application was rebuilt against the verified GCC 12/aarch64 Debian sysroot and checked on the actual phone. Its dynamic dependencies do not include SNPE or FastRPC. The ncnn patch's opt-in FP32 Gemm SLM64 path remains enabled; a separate experiment setting all default shader subgroups to 64 was not selected.

## Prepare crops and run

On the host, prepare already cropped text regions:

```bash
python scripts/prepare_inputs.py /path/to/crops /path/to/raw \
  --device-prefix /path/on/phone/raw
```

Transfer the repository, GPU assets, aarch64 build and prepared inputs to the phone. Run there:

```bash
bash run-gpu.sh /path/on/phone/raw/input.tsv /path/to/result.jsonl 310
```

Each invocation accepts 1–310 crops. Override `VK_DRIVER_FILES` for a different compatible ICD; `SCV41_GPU_ASSETS` and `SCV41_GPU_BINARY` override the model and executable paths. The script defaults to the tested phone's KGSL Mesa installation. Only SCV41 is validated; this is not a flashing or driver installer.

The runner is a batch inference/benchmark CLI, with two six-second idle intervals around the measured work. Throughput and energy exclude initialization and these intervals. It does not install a desktop service or change the existing OCR service, RustDesk, clocks, charging or thermal policy.

## What was optimized

| Stage | Selected implementation |
|---|---|
| CNN | Native PNNX graph; FP16 packed storage and FP16 arithmetic |
| Sequence/classifier tail | FP32 arithmetic and storage; native LayerNorm and attention |
| CNN → tail | GPU cast/packing into FP32, within the same Vulkan command stream; zero feature host readbacks |
| CTC | GPU argmax and probability reduction; 80 index/probability pairs, 640 bytes returned |
| Scheduling | Two GPU workers; at most ten crops in flight; batch size one |

Native operators allow ncnn to select its matching Vulkan kernels, but this does not mean every adjacent operator becomes one shader. The result combines graph, kernel, precision and transfer changes; it is not an isolated measurement of PNNX alone. Model execution has no CPU fallback; CPU input conversion, scheduling and text assembly still occur.

A [three-crop diagnostic layer profile](../benchmarks/pnnx-layer-profile.json) measured the classifier Gemm at about 9.62 ms and several large convolutions at about 6.1 ms each. That diagnostic deliberately used host readbacks to expose ncnn GPU timestamps. Its kernel timings must not be added together to infer production pipeline throughput.

The selected FP16 CNN retained every previously correct line in both evaluation corpora. The 310-crop set improved from 256 to 257 correct; auxiliary 100 stayed at 87. Auxiliary character edits increased from 29 to 30, while the main set decreased from 836 to 835. Both corpora have engineering/calibration exposure. These are regression checks, not evidence of universal or unseen-data losslessness.
