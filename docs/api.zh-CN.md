# Docker GPU API

[首页](../README.zh-CN.md) · [English](api.md) | **简体中文**

v0.3 将 v0.2 的 PNNX 模型做成常驻 HTTP 服务。一个 API 进程管理一个原生推理进程，内部两个 GPU 工作线程；每个进程生命周期内只初始化一次模型。图片通过内存中的二进制管道传递，CNN 特征留在 GPU，CTC 仅回传 640 字节。每次请求无需重新加载模型，也不写图片临时文件，禁止 CPU 推理回退。

**任务范围：已经裁切好的文字行识别。** API 不包含整页检测、版面分析或相机采集。本版使用 **Adreno Vulkan** 加速，不封装旧 SNPE NPU 配置。目前只实机验证了 SCV41、原生 Debian 12 和对应的 KGSL Mesa 环境。

## 在 ARM64 手机上快速部署

手机须已有 Docker 和可用的 KGSL Vulkan 驱动；镜像不包含驱动或固件。可以复用 v0.2 GPU 模型，也可以单独下载。在克隆的本仓库内运行：

```bash
mkdir -p downloads/models downloads/api
gh release download v0.2.0 -R q1820926174-cpu/snapdragon855-ocr-mixed \
  -p sm8150-pnnx-gpu-v0.2.0.tar.gz -p SHA256SUMS -D downloads/models
(cd downloads/models && sha256sum -c SHA256SUMS)
tar -xzf downloads/models/sm8150-pnnx-gpu-v0.2.0.tar.gz
gh release download v0.3.0 -R q1820926174-cpu/snapdragon855-ocr-mixed \
  -p snapdragon855-ocr-api-v0.3.0-linux-arm64.tar.gz -p SHA256SUMS -D downloads/api
(cd downloads/api && sha256sum -c SHA256SUMS)
docker load -i downloads/api/snapdragon855-ocr-api-v0.3.0-linux-arm64.tar.gz
export VIDEO_GID=$(stat -c '%g' /dev/kgsl-3d0)
docker compose up -d --no-build
curl --fail http://127.0.0.1:18081/readyz
```

安装的是 Compose v1 时，将 `docker compose` 换成 `docker-compose -f compose.yaml`。旧手机下游内核可能需要事先配置兼容的 OCI 运行时，实测手机已具备。本仓库不会改动 Docker 系统运行时或设备权限。

Compose 默认只映射本机 18081 端口，使用 UID 10001 和宿主 GPU 用户组，模型及 Mesa 只读挂载，移除额外 capabilities，根文件系统只读。仅传入 `/dev/kgsl-3d0`，无需 privileged、NPU/ION 设备或 Docker socket。要从连接手机的电脑调用，可将 `OCR_BIND_ADDRESS` 设成手机 USB 地址；开放到其他网络时，可通过 `OCR_API_KEY` 配置 Bearer 密钥并随请求发送。

## 调用方法

直接发送**编码后的图片字节**，不使用 multipart 表单：

```bash
curl --fail http://127.0.0.1:18081/v1/recognize \
  -H 'Content-Type: image/png' --data-binary @crop.png
python examples/client.py --url http://127.0.0.1:18081 --concurrency 10 crop1.jpg crop2.png
```

返回字段包括 `text`、`confidence`、`hash`、`gpu_ms`、`total_ms`、`backend` 和 `model`。`total_ms` 包括服务端排队和预处理；客户端时延还包括传输。交互文档和接口定义分别位于 `/docs`、`/openapi.json`，响应使用固定英文键名和 Unicode 文字。

按输入顺序返回的批量调用：

```bash
python - <<'PY'
import base64,json
from pathlib import Path
data={'images':[{'id':'first','image_base64':base64.b64encode(Path('crop.png').read_bytes()).decode()}]}
Path('batch.json').write_text(json.dumps(data))
PY
curl --fail http://127.0.0.1:18081/v1/recognize/batch \
  -H 'Content-Type: application/json' --data-binary @batch.json
```

配置 `OCR_API_KEY` 后，请携带 `Authorization: Bearer <你的密钥>`；示例客户端会读取这个环境变量。批量图片需要纯 base64，不带 data URL 前缀。会应用 EXIF 方向；高度大于宽度 1.5 倍的裁剪图顺时针旋转 90°，与基准测试一致。单张可用 `?auto_rotate=false` 关闭，批量可设置 `"auto_rotate":false`。其余预处理沿用归一化 BGR、48×640 的输入约定。

| 接口 | 用途 |
|---|---|
| `POST /v1/recognize` | 单张编码图片，支持 PNG/JPEG/WebP 等 Pillow 格式 |
| `POST /v1/recognize/batch` | 1–32 张 base64 图片，结果保持输入顺序 |
| `GET /v1/models` | 模型、权重版本和硬件运行方式 |
| `GET /v1/stats` | 排队／完成／失败／拒绝的图片任务计数与模型初始化信息 |
| `GET /metrics` | Prometheus 文本计数器及排队数量 |
| `GET /healthz`、`/readyz` | 进程存活与 GPU 就绪状态 |

## 容量与恢复

默认两个 GPU 工作线程、10 张在途，含等待任务共最多接纳 32 张图片，所有调用者共用这些额度。Uvicorn 只运行一个进程；10 个 HTTP 调用者不会创建 10 套模型。单张限制 5 MiB／2000 万像素，请求体限制 16 MiB。服务不保存或记录图片。

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `OCR_GPU_WORKERS` | 2 | 原生 GPU 工作线程，1–4 |
| `OCR_MAX_INFLIGHT` | 10 | 在途数量，下限为工作线程数，上限 32 |
| `OCR_MAX_PENDING` | 32 | 接纳总量，下限为在途数量，上限 128 |
| `OCR_REQUEST_TIMEOUT` | 30 | 排队＋预处理＋推理总超时，1–300 秒 |
| `OCR_API_KEY` | 空 | 推理、模型信息及监控接口的可选 Bearer 鉴权 |

Compose 示例显式设置了线程和队列默认值，调优时修改对应 environment。队列满返回 **429 + Retry-After**，无效图片 **422**，图片或请求体超限 **413**，GPU 不可用 **503**，推理超时 **504**。批量请求按整批额度接纳或拒绝；客户端断开后，已送入 GPU 的工作仍占用额度直到完成。原生工作进程异常退出时，API 随之退出，由 Docker 重启策略重新加载服务。这是应用进程级恢复，不覆盖内核挂死或断电。

## 从开源源码构建

在 ARM64 Linux 构建主机上：

```bash
docker build -t snapdragon855-ocr-api:0.3.0 .
```

其他架构可用 Buildx，加入 `--platform linux/arm64 --load`。默认 Dockerfile 从源码编译固定版本 ncnn、glslang 及原生程序。也提供复用验证缓存／交叉编译的路径：

```bash
cmake --build build --target pnnx-gpu-worker -j2
docker build -f docker/runtime.Dockerfile -t snapdragon855-ocr-api:0.3.0 .
```

先按 [GPU 指南](gpu-native.zh-CN.md)配置 `build/`，交叉编译时使用 aarch64 工具链。运行镜像的目标架构必须与程序一致。发布的 ARM64 镜像使用经过验证的缓存／交叉编译路径，并在实机运行；CI 另在 amd64 上编译默认源码 Dockerfile，该检查不代表 amd64 GPU 已适配。

## 验证与许可

<!-- api-measurement -->
实测：**6.250 张/秒**，10 并发，客户端 p50 **1599 毫秒**、p95 **1603 毫秒**。这些时延包含 10 并发下的排队，不是单请求 GPU 独立耗时。主动结束工作进程后的自动重启及恢复后真实 GPU 推理已通过。
<!-- /api-measurement -->

[API 基准数据](../benchmarks/api-v0.3.json)来自真实 USB 连接上的 HTTP 请求，包含图片解码和预处理。310 张文字和输出哈希均与 v0.2 一致，其中正确 257 张。这是有工程使用经历的回归集，不是未见过的最终准确率测试。CLI 与 API 分别测量，较小的吞吐差异属于波动，不宣称 API 反而加速了模型。

详见[许可说明](licensing.zh-CN.md)及镜像中的 `/app/licenses`。GPU 程序、ncnn、glslang 均为开源组件；Debian／Python 包保留各自许可。模型使用未改变的 v0.2 开放模型包，单独挂载。镜像不包含高通 SDK／运行库／固件或已编译 HTA 缓存。API 按 FastAPI 的 [lifespan 方式](https://fastapi.tiangolo.com/advanced/events/)常驻模型；设备和只读挂载采用[标准 Docker 配置](https://docs.docker.com/engine/containers/run/)。
