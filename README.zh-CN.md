# 骁龙 855 混合精度 OCR

[English](README.md) | **简体中文**

在运行原生 Linux 的三星 Galaxy S10 SCV41 上，让 **HTA/NPU 与 Adreno GPU 协同执行 PP-OCRv6 文字识别**。本项目公开实验使用的集成代码、模型权重和构建方法，方便复现与继续优化。

**项目处于实验阶段。目前在这台手机上，日常使用仍推荐已有纯 GPU 方案。** 混合配置用于研究精度、吞吐和能耗之间的取舍。

[下载模型](https://github.com/q1820926174-cpu/snapdragon855-ocr-mixed/releases/tag/v0.1.0) · [构建指南](docs/build.zh-CN.md) · [详细测试](docs/benchmarks.zh-CN.md) · [模型说明](docs/model-card.zh-CN.md)

## 实测表现

以下结果来自 **310 张已经裁剪好的文字图片**，输入固定为 48×640，统计区间不含模型初始化。

| 配置 | 正确数 / 310 | 吞吐：条/秒 ↑ | 每条能量：焦耳 ↓ |
|---|---:|---:|---:|
| CPU 基准 | 255 | 2.887 | 1.750 |
| **已有纯 GPU** | **256** | **4.584** | **0.991** |
| 混合 `quality` | 255 | 2.428 | 1.618 |
| 混合 `fast` | 252 | 5.305 | 0.736 |

`quality` 保住了 CPU 原本识别正确的全部样本，但比纯 GPU 更慢、每条耗能更多。`fast` 更快、更省电，同时丢失了 5 个 CPU 原本正确的样本；因为另有 2 个样本识别变好，最终正确数只减少了 3 个。

能量数据来自**包含桌面后台的整机供电估算**，各批次温度不同。相对 CPU，`quality` 约降低 7.5% 的每条能量，这个小幅差距还需要同温复测；`fast` 约降低 58%，但伴随精度退步。详见[测量方法与限制](docs/benchmarks.zh-CN.md)。

这组图片是开发过程中使用的回归集，不能据此保证量化模型在所有新图片上无损。

## 运行条件

| 项目 | 已验证环境 |
|---|---|
| 手机 | 三星 SCV41，骁龙 855 / SM8150，Adreno 640 |
| 手机系统 | 原生 Debian 12，NPU、ION、FastRPC 和 KGSL Mesa 已适配 |
| 外部软件 | 使用者自行合法获取的 SNPE 1.59.0.3230 SDK/运行库及匹配的设备固件 |
| 准备电脑 | x86-64 Linux 或 WSL；预处理和模型检查使用 Python 3.11；SDK 使用兼容的 Python 3.6 环境 |

公开包提供集成源码和开放模型参数。高通 SDK、运行库及固件需要另行获取，HTA 容器需要在本地生成。目前只验证了 SCV41；刷机和系统驱动安装不在本项目范围内。

## 开始使用

### 1. 下载并校验模型

在准备电脑上执行，需要先安装 Git 和 GitHub CLI：

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

模型文件包含分区 ONNX、冻结的 W8 权重网格、量化参数、GPU ncnn 模型、通道缩放与稀疏修正、字典及参考 FP32 模型。各部分的作用见[模型说明](docs/model-card.zh-CN.md)。

### 2. 构建 HTA 容器和执行器

按照[构建指南](docs/build.zh-CN.md)完成以下步骤：

1. 在 x86-64 电脑上，用自己的 SDK 生成 SM8150 HTA 容器，并核对权重和量化参数。
2. 使用指定版本的 ncnn 和补丁，构建 aarch64 执行器。
3. 检查手机的平台依赖，将准备好的文件传入手机。

### 3. 准备文字裁剪图，在手机上执行

先在电脑上将已经裁剪好的文字图片转换成模型输入：

```bash
python scripts/prepare_inputs.py /path/to/crops /path/to/raw \
  --device-prefix /path/on/phone/raw
```

将生成的 raw 文件和 `input.tsv` 复制到手机。随后在**手机上**进入准备好的仓库目录执行：

```bash
sudo bash run.sh quality /path/on/phone/raw/input.tsv /path/to/result.jsonl 310

# 可选：速度更高，但存在已知精度退步。
sudo bash run.sh fast /path/on/phone/raw/input.tsv /path/to/result-fast.jsonl 310
```

模型负责裁剪区域内的文字识别；整页检测、方向判断和版面分析需自行接入。每次运行接受 1–310 条输入。运行器持有 NPU 独占锁，超时为 170 秒，退出时恢复此前的 NPU 固件供电状态。

## 两种配置如何分工

| 配置 | NPU 负责 | GPU 负责 |
|---|---|---|
| `quality` | 12 个大卷积采用 W8A8；每层执行两次，细化输入量化残差 | 其余算子与存储保持 FP32，并修正稀疏权重离群值 |
| `fast` | 16 个大卷积采用 W8A8；每层执行一次 | CNN 使用 FP16 打包存储、FP32 算术；分类头保持 FP32，并融合数据转换 |

这些卷积分别约占原始卷积乘加运算量的 54.9% 和 75.5%。单进程内的 4 个处理线程共享 12 或 16 个 NPU 上下文，在途请求最多为 10 个，禁止回退到 CPU。**两次 INT8 运算不代表硬件原生支持 W8A16。**

## 文档与验证

| 内容 | 简体中文 | English |
|---|---|---|
| 构建与平台依赖 | [构建指南](docs/build.zh-CN.md) | [Build guide](docs/build.md) |
| 测试结果与能耗解读 | [测试与能耗](docs/benchmarks.zh-CN.md) | [Benchmarks](docs/benchmarks.md) |
| 模型设计与评估边界 | [模型说明](docs/model-card.zh-CN.md) | [Model card](docs/model-card.md) |
| 许可与发布内容 | [许可与发布范围](docs/licensing.zh-CN.md) | [Licensing](docs/licensing.md) |

原交付包的两个配置，以及通过公开源码和权重重建的两个配置，都完成了全量 310 图复查，识别文本与原结果一致。详见[复查记录](benchmarks/recheck.json)和[测试数据](benchmarks/results.json)。CI 检查源码及输入约定；硬件验证在真实手机上进行。

自有集成代码与 PaddleOCR 衍生模型采用 Apache-2.0，ncnn 补丁保留 BSD-3-Clause 署名。详见 [NOTICE](NOTICE) 和 [licenses/](licenses/)。高通依赖仍适用其独立许可。
