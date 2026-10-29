# 构建与复现

[返回首页](../README.zh-CN.md) · [English](build.md) | **简体中文**

构建分为两个阶段：先在 x86-64 电脑上生成 NPU 模型，再为 aarch64 手机构建执行器。运行前，手机需要已经具备可用的底层驱动与运行环境。

**新的纯 GPU 路径请先阅读 [PNNX GPU 指南](gpu-native.zh-CN.md)。** 本页记录旧混合 NPU/GPU 配置。只构建 GPU 程序不需要 SNPE。

## 先检查准备条件

| 在哪里 | 需要什么 |
|---|---|
| x86-64 Linux / WSL 电脑 | 自行合法获取的 SNPE **1.59.0.3230** SDK 及其兼容的 Python 3.6 环境 |
| 预处理与完整模型检查 | Python 3.11，安装仓库中的 `requirements.txt` |
| aarch64 执行器构建环境 | GCC 12、CMake、Ninja、指定版本的 ncnn，以及 aarch64 SNPE 运行库 |
| 手机 | 原生 Debian 12，NPU、ION、CDSP/FastRPC 和 KGSL Mesa 已正常工作 |

仓库不提供或镜像高通 SDK。旧 SDK 的 Python 环境需要单独安装它所支持的 ONNX、numpy 和转换器依赖；`build_dlc.py` 这个构建包装脚本本身只需要现代 Python 的标准库。

## 1. 在电脑上生成 HTA 容器

下载、解压并校验 Release 中的开放模型后，执行：

```bash
python scripts/build_dlc.py \
  --sdk /path/to/snpe-1.59.0.3230 \
  --sdk-python /path/to/sdk-python36/bin/python \
  --assets "$PWD/models" --work /tmp/sm8150-dlc-build
```

脚本通过 `--quantization_overrides` **和 `--override_params`** 保留冻结的 W8 权重网格与输入、输出量化参数。全零占位输入只用于满足 SDK 的输入列表接口，不会重新校准这些参数。

每层转换后都会检查以下项目，通过后才将 DLC 复制到 `models/npu/n_*/`：

- 输入、输出、权重的量化步长、偏移和位宽与导出值一致。
- 实际反量化权重与目标值的差异小于权重量化步长的 0.01 倍。
- 编译目标为 HTA，且没有引入 Permute。

编译使用 `--enable_hta --hta_partitions sm8150:0-1 --bias_bitwidth 32`。较新的 HTP SDK/缓存不能直接替代这代 HTA 环境。虽然旧 SDK 能编译 W8A16，但手机现有运行时拒绝了它，因此本项目没有采用原生 W8A16。

## 2. 构建 aarch64 执行器

可以在手机上原生编译，也可以使用 aarch64 Debian sysroot 交叉编译。已验证的环境使用 GCC 12，以及 ncnn 提交 `9fc8106e6634a60f39945d63e5ed2dd0704ac9fc`。

先获取指定版本并应用补丁：

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
  -DBUILD_MIXED=ON \
  -Dncnn_DIR="$PWD/_deps/ncnn-install/lib/cmake/ncnn" \
  -DSNPE_ROOT=/path/to/authorized-sdk \
  -DSNPE_LIBRARY=/path/to/aarch64/libSNPE.so
cmake --build build -j4
```

交叉编译时，给**两条 CMake 配置命令**都加上 `-DCMAKE_TOOLCHAIN_FILE=/path/to/your-aarch64-toolchain.cmake`。目标运行库必须是 aarch64 版本。已有构建缓存可以复用，前提是 ncnn 提交、补丁和构建选项一致。

ncnn 补丁保留了这套已验证环境中的 Turnip FP16 能力，增加可选的 FP32 SLM64 Gemm，并支持记录、拒绝 GPU 算子回退到 CPU。着色器子组大小必须先于本地工作组大小设置。这是针对已验证设备的补丁，不应当作通用 Mesa 修复使用。

## 3. 核对手机平台依赖

运行器沿用已适配 SCV41 的目录布局：

| 依赖 | 预期位置或状态 |
|---|---|
| NPU 设备 | `/dev/msm_npu`，NPU/ION 与 CDSP/FastRPC 接口可用 |
| SNPE 与设备固件 | `/opt/scv41-snpe-1.59.0.3230`，包含匹配的 `fastrpc-v4/libcdsprpc.so`、运行库及使用者合法获取的 DSP 固件 |
| KGSL Mesa Vulkan ICD | `/opt/scv41-mesa-kgsl/share/vulkan/icd.d/freedreno_icd.armv8-a.json` |
| 已验证的非缓存 ION 辅助库 | `/opt/scv41-npu-test/libnpu-uncached-ion.so` |

ION 辅助库的源码位于 `runtime/npu-uncached-ion.c`，CMake 会构建它，但不会自动安装或替换系统中的辅助库。

以上是**手机运行环境的前提条件**，不随公开包分发。项目不会解锁、刷机、安装厂商固件，也不修改内核超时、频率、充电或温控策略。其他设备需要另行适配与验证。准确路径及 NPU 管理逻辑见 `run.sh` 和 `runtime/npu-manager.py`。

## 4. 检查构建结果

在电脑上执行：

```bash
python -m unittest discover -s tests -v
bash -n run.sh
python scripts/verify_models.py models
```

硬件验收需要真实手机及已适配的平台环境。CI 只检查源码与输入约定，不模拟 HTA 硬件。
