# PNNX 原生 GPU 导出与运行

[首页](../README.zh-CN.md) · [English](gpu-native.md) | **简体中文**

v0.2 GPU 路径从原始 PP-OCRv6 模型重新导出，没有复用为旧 SNPE 转换器展开的兼容图。在已验证的 Adreno 640 上，模型通过 ncnn Vulkan 执行。**不需要 SNPE、NPU 运行库或 root。** 手机仍须已有可用的 Vulkan 驱动和 GPU 访问权限。

## 下载与校验

在准备主机上，进入克隆的本仓库后运行：

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

这个独立模型包包含原始 ONNX、CNN／尾部／完整图的 ncnn 文件、字典、模型约定和文件哈希。v0.1 混合模型包仍单独保留。不包含高通文件、可执行程序、测试图片或逐图文字。

## 复现转换

可以直接使用随包提供的图。如需重新导出，安装固定版本的 PNNX 二进制包，跳过其可选 Torch 前端依赖，然后运行：

```bash
pip install --no-deps -r requirements-gpu-export.txt
python scripts/export_gpu_pnnx.py \
  --source models/gpu-native/source/original.onnx \
  --output /tmp/ocr-pnnx-export
```

导出脚本检查原始模型的 SHA-256，将输入固定为 `[1,3,48,640]`，仅为 GPU CTC 移除末端 Softmax，并在原有平均池化输出 `[1,768,1,80]` 处分图。送入 PNNX 前，保留的 ONNX 初始化权重逐字节检查一致；PNNX 随后可能折叠、融合权重。保留原有基于 Erf 的 GELU。完整图含 14 个原生 GELU、5 个 LayerNorm 和 2 个 MultiHeadAttention。ncnn Vulkan 不支持的单维 Squeeze 用等价的固定形状 Reshape 替换。

在 20 张图片上，FP32 转换检查与原始 ONNX Runtime 的每张 80 帧 argmax 全部一致，最大 logit 绝对误差低于 0.000153。这检查的是图转换，**不代表手机所选 FP16 运行配置逐位无损**。详见[精度报告](../benchmarks/pnnx-precision.json)。

[PNNX 官方说明](https://github.com/pnnx/pnnx)介绍了 ONNX 转换和独立二进制工具；[旧 onnx2ncnn 文档](https://github.com/Tencent/ncnn/blob/master/tools/onnx/README.md)建议采用 PNNX 进行新的转换工作。

## 构建手机端程序

按[构建指南第 2 节](build.zh-CN.md#2-构建-aarch64-执行器)构建固定版本、带补丁的 ncnn。构建 GPU 程序时，将该指南中依赖 SNPE 的应用配置命令替换为：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR="$PWD/_deps/ncnn-install/lib/cmake/ncnn" \
  -DBUILD_MIXED=OFF
cmake --build build -j4
```

交叉编译时，ncnn 和应用两处配置都要加入自己的 aarch64 工具链。公开应用源码已用验证过的 GCC 12／aarch64 Debian sysroot 重新构建并在实机检查，其动态依赖不包含 SNPE 或 FastRPC。继续启用 ncnn 补丁中的可选 FP32 Gemm SLM64 路径；另一个将所有默认着色器子组改成 64 的试验没有被选用。

## 准备图片与运行

在主机上准备已经裁切好的文字区域：

```bash
python scripts/prepare_inputs.py /path/to/crops /path/to/raw \
  --device-prefix /path/on/phone/raw
```

将仓库、GPU 模型、aarch64 构建结果和输入传到手机。在手机上运行：

```bash
bash run-gpu.sh /path/on/phone/raw/input.tsv /path/to/result.jsonl 310
```

每次接受 1–310 张图片。其他兼容 ICD 可通过 `VK_DRIVER_FILES` 指定；`SCV41_GPU_ASSETS` 和 `SCV41_GPU_BINARY` 可指定模型和程序目录。脚本默认使用已测试手机的 KGSL Mesa 安装路径。目前只验证了 SCV41；本仓库不负责刷机或驱动安装。

程序是批量推理／测试 CLI，在测量工作前后各有 6 秒空闲间隔。吞吐与能耗排除初始化和这两个间隔。不安装桌面服务，也不改动现有 OCR 服务、RustDesk、频率、充电或温控策略。

## 具体优化了什么

| 环节 | 选用的实现 |
|---|---|
| CNN | PNNX 原生图；FP16 打包存储与 FP16 运算 |
| 序列／分类尾部 | FP32 运算和存储；原生 LayerNorm 与注意力 |
| CNN → 尾部 | 同一 Vulkan 命令流中在 GPU 转换为 FP32 并调整打包，无特征回传到主机 |
| CTC | GPU argmax 与概率归约；回传 80 对索引／概率，共 640 字节 |
| 调度 | 两个 GPU 工作线程；最多 10 张在途；单张批次 |

保留原生算子便于 ncnn 选择对应 Vulkan 内核，但不代表每组相邻算子都融合成一个着色器。结果包含图、内核、精度和数据传输的共同变化，不能归因于 PNNX 单独一个因素。模型执行禁止回退 CPU；输入转换、调度和文字拼装仍由 CPU 完成。

[三张图片的逐层诊断](../benchmarks/pnnx-layer-profile.json)中，分类 Gemm 约 9.62 毫秒，几个大卷积各约 6.1 毫秒。诊断程序特意开启了主机回传，便于读取 ncnn GPU 时间戳；不能把这些内核时间相加来推算正式流水线吞吐。

所选 FP16 CNN 保留了两组图片中旧版识别正确的全部整行。310 张主集从 256 张正确提升到 257 张，100 张补充集保持 87 张正确。补充集字符编辑错误从 29 增加到 30，主集从 836 降至 835。两组数据均有工程调优／校准使用经历；这些是回归检查，不能证明对所有图片或未见数据无损。
