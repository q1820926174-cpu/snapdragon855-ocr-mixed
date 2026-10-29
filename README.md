# Snapdragon 855 mixed-precision OCR

面向 **Samsung Galaxy S10 SCV41 / Snapdragon 855 (SM8150) / Adreno 640 原生 Linux** 的 PP-OCRv6_medium_rec 混合量化实验。大卷积运行在 HTA/NPU，精度敏感部分运行在 Vulkan GPU。

**当前日常使用仍推荐已有纯 GPU 方案。** 本仓库用于混合量化探索；`quality` 是混合模型内部的保精度配置，不代表它比纯 GPU 更快或更省电。没有替换手机原有的纯 GPU 正式服务。

**这是开源集成代码和开放权重，不是完整开源硬件运行栈。** Qualcomm SNPE 1.59、FastRPC、DSP/NPU 固件和 KGSL Mesa 需要使用者自行合法提供。公开包不包含 SDK、SDK 头文件、厂商二进制、预编译 HTA 缓存、刷机镜像或测试图片。只验证了 SCV41，不能据此认为其他 855 设备可以直接使用。

## 实测结果

固定 310 张开发回归文字裁剪图，输入 48×640，4 个处理线程、10 个有界在途请求：

| 配置 | 正确/310 | CPU 原本正确但退步 | 条/秒 | 系统 W | J/条 |
|---|---:|---:|---:|---:|---:|
| CPU 基准 | 255 | 0 | 2.887 | 5.052 | 1.750 |
| 已有纯 GPU | 256 | 0 | 4.584 | 4.544 | 0.991 |
| **quality：保精度** | **255** | **0** | **2.428** | **3.930** | **1.618** |
| fast：速度实验 | 252 | 5 | 5.305 | 3.907 | 0.736 |

- `quality`：12 个大卷积使用 W8A8，覆盖约 54.9% 原始卷积 MAC。每层两次 INT8 计算细化输入量化残差；其余 GPU 计算和存储保持 FP32，GPU 稀疏修正少量权重离群值。**两次 INT8 不是硬件 W8A16。**
- `fast`：16 个大卷积使用 W8A8，覆盖约 75.5% 卷积 MAC；GPU CNN 使用 FP16 打包存储、FP32 算术，分类头保持 FP32。它存在新增错误，不是等精度加速结论。
- NPU 模型在单进程中共享，实际为 12/16 个上下文，禁止 CPU fallback，没有整图重跑回退。

**能耗必须区分配置**：相对 CPU，quality 每条能量约少 7.5%，速度仍慢于 CPU；fast 约少 58%，但有精度退步。quality 比已有纯 GPU 版更费电。整机供电估算包含桌面后台，CPU/GPU 基准批次与温度不同，特别是小幅差距不能当作严格同温能效结论。见 [测试数据与边界](docs/benchmarks.md)。

## 下载开放模型

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

模型包包含分区 ONNX、冻结的 W8 权重网格及量化参数、GPU ncnn 模型、通道缩放/稀疏修正、字典和参考 FP32 ONNX。ONNX 中的 W8 网格用 FP32 常量存放以便导入旧 SDK，**不会让普通 ONNX Runtime 自动执行 NPU INT8**。按照 [构建说明](docs/build.md)，在本地生成 SM8150 HTA DLC 并构建执行器。

## 运行

准备阶段需要 x86-64 Linux/WSL 上的自备 SNPE 1.59 SDK；执行阶段需要已适配的 aarch64 原生 Linux 手机。没有自动刷机、解锁、下载厂商固件或系统驱动安装流程。

```bash
# 本地生成文字裁剪图输入；先自行完成检测/裁剪。
python scripts/prepare_inputs.py /path/to/crops /path/to/raw \
  --device-prefix /path/on/phone/raw
# 将 raw 和 input.tsv 复制到手机，再在手机上执行：
sudo bash run.sh quality /path/on/phone/raw/input.tsv /path/to/result.jsonl 310
# 有精度退步的可选速度配置：
sudo bash run.sh fast /path/on/phone/raw/input.tsv /path/to/result-fast.jsonl 310
```

模型只做文字识别，不含整页检测、方向判断或版面分析。输入为 float32 NHWC BGR [1,48,640,3]，按高 48 缩放、宽不超过 640，`/127.5-1` 后右侧补零。限制每批 1–310 条；平台 NPU 管理器持有独占锁、超时 170 秒，退出恢复原固件供电状态。

## 复查、许可与限制

- [模型说明](docs/model-card.md)、[构建复现](docs/build.md)、[许可证边界](docs/licensing.md)。
- [发布前复查](benchmarks/recheck.json)：重新从原始交付包执行两个完整 310 图配置；发布模型的本地 DLC 重建另有逐层参数/权重审计。
- 这 310 张是开发回归集，不是从未参与开发选择的独立最终测试集；不承诺所有图片无损。测试图片和转录不公开。
- 默认模型来自 PTQ；尝试过的 QAT/扩大激活范围候选未通过精度门槛，没有冒充默认模型成果。
- 自有源码采用 Apache-2.0；PaddleOCR 权重衍生物采用上游 Apache-2.0；ncnn 补丁保留 BSD-3-Clause。见 [NOTICE](NOTICE) 和 [licenses/](licenses/)。闭源依赖不在上述开源许可范围内。

## English summary

Experimental PP-OCRv6 recognition on **native Linux SCV41 / SM8150**, with shared HTA INT8 convolution contexts and Vulkan GPU sensitive operators. The `quality` profile retained all CPU-correct crops on a 310-crop **development regression set**, but was slower than CPU. The `fast` profile increased throughput with five new errors. Power estimates cover the whole system; baseline temperatures differ. This repository releases integration source and model-owned tensors, **not Qualcomm SDK/runtime, firmware or compiled HTA cache**. Build the latter locally with your own authorized SNPE 1.59 installation. No universal 855-device support or lossless-accuracy claim is made.
