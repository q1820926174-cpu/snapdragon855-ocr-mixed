# 骁龙 855 OCR：原生 GPU 与 NPU/GPU 混合推理

[English](README.md) | **简体中文**

在运行原生 Linux 的三星 Galaxy S10 SCV41 上执行 **PP-OCRv6 文字识别**。目前建议使用的实验路径为 **PNNX → ncnn Vulkan → Adreno 640**；此前的 HTA/NPU + GPU 混合配置继续保留，方便对比。

**v0.2：**原生 GPU 版保留了旧版原本正确的全部回归整行，在下述受控测试中，比旧纯 GPU 版吞吐提高 **36.0%**、整机每张能耗降低 **28.3%**。不需要高通 SDK 或 NPU 运行库。

[下载 v0.2 模型](https://github.com/q1820926174-cpu/snapdragon855-ocr-mixed/releases/tag/v0.2.0) · [GPU 构建／导出指南](docs/gpu-native.zh-CN.md) · [测试详情](docs/pnnx-benchmarks.zh-CN.md)

## Docker API（v0.3）

已提供常驻 GPU HTTP 服务：实测手机 **10 并发 6.250 张/秒**，包含上传、解码和预处理。310 张文字及输出哈希全部与 v0.2 一致，模型只初始化一次。支持单张／批量识别、有界队列、可选 Bearer 鉴权、监控，以及工作进程异常后的自动恢复。

[下载 ARM64 Docker 镜像](https://github.com/q1820926174-cpu/snapdragon855-ocr-mixed/releases/tag/v0.3.0) · [部署与 API 指南](docs/api.zh-CN.md) · [API 实测数据](benchmarks/api-v0.3.json)

模型单独挂载，继续使用 v0.2。GPU 加速调用宿主 KGSL Mesa，不打包 SNPE 或 NPU 运行库。这仍是**文字裁剪图识别**，不包含整页检测。仓库同时提供源码 Dockerfile 和已验证的缓存／交叉编译打包方式。

## 实测结果

每种配置运行 **两轮完整的 310 张裁剪图，取中位数**；每次等电池温度在间隔 5 秒的三次检查中均不超过 35.5°C 后开始。第二轮反向排列，使用设备公共互斥锁排除其他受管理的加速测试。数据不含模型初始化。

| 配置 | 正确 / 310 | 张/秒 ↑ | 整机瓦数 | 焦耳/张 ↓ |
|---|---:|---:|---:|---:|
| 旧纯 GPU | 256 | 4.591 | 4.344 | 0.946 |
| **PNNX 原生 GPU** | 257 | 6.243 | 4.233 | 0.678 |
| 混合 `quality` | 255 | 2.463 | 3.697 | 1.502 |
| 混合 `fast` | 252 | 5.243 | 3.727 | 0.711 |

原生 GPU 保留了旧 GPU 正确的全部 256 张，以及 CPU 正确的全部 255 张。混合 `quality` 保留 CPU 正确样本，但速度更慢；`fast` 丢失 5 张 CPU 正确样本，不属于同等精度方案。另有 **1 次混合运行库测试未完成**，已单独记录，未计入性能中位数。

能耗来自 **ISYS × VSYS 整机供电轨估算**，包含桌面，不是 GPU 单独功耗或电池放电能量。两轮复测只说明这些运行结果，不能代表跨设备的统计确定性。提供[相对时间功耗轨迹和汇总](benchmarks/pnnx-controlled.json)。历史 CPU 数据保留在[早期报告](docs/benchmarks.zh-CN.md)，本轮受控对比未重测 CPU。

原生 GPU 两轮能耗为 0.641–0.716 焦耳/张，与混合 `fast` 的范围重叠；不能凭较小的中位数差距断言其能耗稳定优于这个精度较低的配置。

这两组数据有开发使用经历，不是未见过的独立最终测试集。补充 100 张图片保留了旧 GPU 正确的全部 87 张，但字符错误增加 1 个；主集字符编辑错误减少 1 个。**不宣称 FP16 普遍无损。**详见[精度报告](benchmarks/pnnx-precision.json)。

## 运行 GPU 版

在装有 Git 和 GitHub CLI 的 x86-64 Linux／WSL 准备主机上运行：

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

按 [GPU 指南](docs/gpu-native.zh-CN.md)，使用固定版本、带补丁的 ncnn 构建 aarch64 程序。手机须已有可用 Vulkan 驱动及 GPU 访问权限。仅验证了 SCV41／原生 Debian 12 与实测 KGSL Mesa 环境；本仓库不负责驱动安装或刷机。

在主机上准备已经裁切成文字区域的图片：

```bash
python scripts/prepare_inputs.py /path/to/crops /path/to/raw \
  --device-prefix /path/on/phone/raw
```

将输入、仓库、GPU 模型与 aarch64 构建结果传到手机，**在手机上**运行：

```bash
bash run-gpu.sh /path/on/phone/raw/input.tsv /path/to/result.jsonl 310
```

这是裁剪文字识别，不是整页 OCR。批量 CLI 每次接受 1–310 张图片，使用两个工作线程、最多 10 张在途。不安装服务，也不改动 RustDesk 或桌面。

## GPU 路径改了什么

旧 GPU 路径复用了为 SNPE 展开的兼容图。新版从原始 ONNX 出发，用 PNNX 20261020 保留原生 GELU、LayerNorm 和注意力算子。CNN 存储和运算使用 FP16，序列／分类尾部保持 FP32。CNN 特征留在 GPU，融合 CTC 每张仅回传 640 字节。模型执行禁止回退 CPU。提升来自图、内核、精度和传输方式的共同变化，不能只归因于 PNNX。

模型包包含原始 ONNX、图哈希和独立导出脚本对应的模型。公开源码重建的图已逐文件核对，公开构建程序及脚本已回放完整 310 张。详见[复现检查](benchmarks/pnnx-recheck.json)。

## 旧混合配置

保留独立的 [v0.1 模型包](https://github.com/q1820926174-cpu/snapdragon855-ocr-mixed/releases/tag/v0.1.0)和[混合构建指南](docs/build.zh-CN.md)。这两个配置需要自行合法获取 SNPE 1.59 SDK／运行库，并具备兼容的 NPU／ION／FastRPC／固件环境。以 `-DBUILD_MIXED=ON` 构建，在本地生成 HTA 容器后，在已验证手机上运行：

```bash
sudo bash run.sh quality /path/on/phone/raw/input.tsv /path/to/result-quality.jsonl 310
sudo bash run.sh fast /path/on/phone/raw/input.tsv /path/to/result-fast.jsonl 310
```

`quality` 共用 12 个 NPU 上下文，每个选中卷积执行两次 INT8，其余 GPU 使用 FP32；`fast` 使用 16 个上下文，每层一次 INT8，CNN 用 FP16 打包存储和 FP32 运算。两者均使用四个工作线程、最多 10 张在途。两次 INT8 不代表硬件原生 W8A16。原有发布文件未被替换。

## 文档

| Read about / 内容 | English | 简体中文 |
|---|---|---|
| Native GPU export/build / 原生 GPU 导出与构建 | [GPU guide](docs/gpu-native.md) | [GPU 指南](docs/gpu-native.zh-CN.md) |
| Controlled tests / 受控测试 | [PNNX benchmarks](docs/pnnx-benchmarks.md) | [PNNX 测试](docs/pnnx-benchmarks.zh-CN.md) |
| Mixed build / 混合版构建 | [Build guide](docs/build.md) | [构建指南](docs/build.zh-CN.md) |
| Original measurements / 早期测量 | [Legacy benchmarks](docs/benchmarks.md) | [早期测试](docs/benchmarks.zh-CN.md) |
| Model and licenses / 模型与许可 | [Model card](docs/model-card.md) · [Licensing](docs/licensing.md) | [模型说明](docs/model-card.zh-CN.md) · [许可](docs/licensing.zh-CN.md) |

自有集成源码与 PaddleOCR 衍生模型采用 Apache-2.0，ncnn 补丁保留 BSD-3-Clause 署名。详见 [NOTICE](NOTICE) 与 [licenses/](licenses/)。不分发高通软件与固件、已编译 HTA 缓存、可执行程序、测试图片／文字或私有设备访问文件。
