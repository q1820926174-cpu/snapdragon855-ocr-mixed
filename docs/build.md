# Build and reproduce

[Home](../README.md) · **English** | [简体中文](build.zh-CN.md)

Prepare the HTA containers on an x86-64 host, then build the executors for the aarch64 phone. The phone's underlying drivers and runtime must already work.

## Before you start

| Where | Required tools |
|---|---|
| x86-64 Linux / WSL host | Your authorized SNPE **1.59.0.3230** SDK and its compatible Python 3.6 environment |
| Preprocessing and full asset checks | Python 3.11 with `requirements.txt` |
| aarch64 build environment | GCC 12, CMake, Ninja, the pinned ncnn version and an aarch64 SNPE library |
| Phone | Native Debian 12 with working NPU, ION, CDSP/FastRPC and KGSL Mesa |

## 1. Model preparation on an x86-64 Linux host

Use an authorized **SNPE 1.59.0.3230** SDK and the SDK's compatible Python 3.6 environment. The public repository does not install or mirror that SDK. Host preprocessing and full asset verification use Python 3.11 with `requirements.txt`; the old SDK environment separately needs its own supported ONNX/numpy/converter dependencies. The build wrapper itself only needs a modern standard-library Python environment.

After extracting the release asset and verifying it:

```bash
python scripts/build_dlc.py \
  --sdk /path/to/snpe-1.59.0.3230 \
  --sdk-python /path/to/sdk-python36/bin/python \
  --assets "$PWD/models" --work /tmp/sm8150-dlc-build
```

The script preserves the exported W8 weight grid and fixed input/output encodings with `--quantization_overrides` **and `--override_params`**. A zero placeholder input only satisfies the SDK input-list interface; it does not recalibrate the frozen quantization parameters.

Every layer is audited after conversion. Before its DLC is copied into `models/npu/n_*/`:

- Input/output/weight scales, offsets and bitwidth must match the exported values.
- Actual dequantized weights must match within 0.01 weight LSB.
- The compiled target must be HTA without a Permute.

The DLC uses `--enable_hta --hta_partitions sm8150:0-1 --bias_bitwidth 32`. A modern HTP SDK/cache is not a drop-in replacement for this 2019-generation HTA runtime. Native W8A16 compiled with this SDK was rejected by the installed phone runtime and is not used.

## 2. Build the aarch64 runtime

Build natively on the phone or cross-compile against your aarch64 Debian sysroot. The tested stack used GCC 12 and ncnn commit `9fc8106e6634a60f39945d63e5ed2dd0704ac9fc`, with the supplied opt-in patch:

```bash
git clone https://github.com/Tencent/ncnn.git _deps/ncnn
git -C _deps/ncnn checkout 9fc8106e6634a60f39945d63e5ed2dd0704ac9fc
git -C _deps/ncnn submodule update --init --recursive
git -C _deps/ncnn apply ../../patches/ncnn-sm8150.patch

cmake -S _deps/ncnn -B _deps/ncnn-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/_deps/ncnn-install" \
  -DNCNN_VULKAN=ON -DNCNN_SHARED_LIB=OFF -DNCNN_BENCHMARK=OFF \
  -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
  -DCMAKE_C_FLAGS='-fno-tree-loop-vectorize -fno-tree-slp-vectorize' \
  -DCMAKE_CXX_FLAGS='-fno-tree-loop-vectorize -fno-tree-slp-vectorize'
cmake --build _deps/ncnn-build -j4
cmake --install _deps/ncnn-build

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR="$PWD/_deps/ncnn-install/lib/cmake/ncnn" \
  -DSNPE_ROOT=/path/to/authorized-sdk \
  -DSNPE_LIBRARY=/path/to/aarch64/libSNPE.so
cmake --build build -j4
```

For cross compilation, add `-DCMAKE_TOOLCHAIN_FILE=/path/to/your-aarch64-toolchain.cmake` to **both** configure commands. Do not point them at x86-64 runtime libraries. Existing verified build caches can be reused if the ncnn commit, patch and options match.

The ncnn patch retains the upstream Turnip FP16 capability on this qualified stack, adds opt-in FP32 SLM64 Gemm, and supports strict logging/rejection of GPU-to-CPU fallback. The shader subgroup size must be set before local workgroup size. It is a device-specific patch; it is not a general Mesa compatibility fix.

## 3. Qualified phone platform prerequisites

The runner expects the existing SCV41 platform layout:

- `/dev/msm_npu` and working downstream NPU/ION + CDSP/FastRPC interfaces.
- Your SNPE installation under `/opt/scv41-snpe-1.59.0.3230`, including its compatible `fastrpc-v4/libcdsprpc.so`, runtime and your device's authorized DSP firmware.
- Working KGSL Mesa Vulkan ICD `/opt/scv41-mesa-kgsl/share/vulkan/icd.d/freedreno_icd.armv8-a.json`.
- The tested uncached ION helper `/opt/scv41-npu-test/libnpu-uncached-ion.so`. Its original source is included as `runtime/npu-uncached-ion.c`; CMake builds it, but does not install or replace any system helper automatically.

These are **platform prerequisites, not files supplied by the open release**. The code does not unlock, flash, install proprietary firmware, change kernel timeouts, clocks, charging or thermal policy. Porting another device requires separate validation. See `run.sh` and `runtime/npu-manager.py` for the exact paths and runtime ownership boundary.

## 4. Check the build

Run on the host:

```bash
python -m unittest discover -s tests -v
bash -n run.sh
python scripts/verify_models.py models
```

Physical tests need the real device and qualified platform. CI checks source and input contracts; it does not emulate HTA hardware.
