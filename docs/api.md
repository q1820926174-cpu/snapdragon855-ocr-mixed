# Docker GPU API

[Home](../README.md) · **English** | [简体中文](api.zh-CN.md)

v0.3 turns the v0.2 PNNX model into a resident HTTP service. A single API process owns one native worker process with two GPU inference workers. Models load once per process lifetime; images pass through a binary pipe in memory. CNN features stay on GPU, and CTC returns 640 bytes. No per-request model loading, image files or CPU inference fallback.

**Task: recognition of already cropped text lines.** This API does not perform page detection, layout analysis or camera capture. This release accelerates inference on **Adreno Vulkan**, not the legacy SNPE NPU profiles. Only SCV41, native Debian 12 and the tested KGSL Mesa stack are hardware-qualified.

## Quick start on the ARM64 phone

Docker and a working host KGSL Vulkan driver must already be installed. The driver/firmware are not in the image. Reuse your v0.2 GPU assets or download them separately. From a clone of this repository:

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

On installations with Compose v1, replace `docker compose` with `docker-compose -f compose.yaml`. Old downstream phone kernels may require an already configured compatible OCI runtime, as on the tested phone. This repository does not change Docker's system runtime or device permissions.

The supplied Compose configuration publishes localhost port 18081, runs as UID 10001 with the host GPU group, mounts models and Mesa read-only, drops capabilities and uses a read-only root filesystem. Only `/dev/kgsl-3d0` is passed through; privileged mode, NPU/ION device access and a Docker socket mount are not required. Set `OCR_BIND_ADDRESS` to the phone's USB address for access from its connected computer. For access from another network, set a bearer token through `OCR_API_KEY` and send it with the requests.

## Calls

Send the **encoded image bytes**, not multipart form data:

```bash
curl --fail http://127.0.0.1:18081/v1/recognize \
  -H 'Content-Type: image/png' --data-binary @crop.png
python examples/client.py --url http://127.0.0.1:18081 --concurrency 10 crop1.jpg crop2.png
```

Response fields include `text`, `confidence`, `hash`, `gpu_ms`, `total_ms`, `backend` and `model`. `total_ms` includes server queuing/preprocessing; client latency also includes transfer. API docs and schema are at `/docs` and `/openapi.json`; responses use stable English field names and Unicode text.

For an ordered batch:

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

Use `Authorization: Bearer <your-token>` when `OCR_API_KEY` is configured; the example client reads that environment variable. Pure base64 is expected, without a data-URL prefix. EXIF orientation is applied; crops taller than 1.5 times their width rotate 90° clockwise, matching the benchmark. Disable that heuristic with `?auto_rotate=false` for the single endpoint or `"auto_rotate":false` for a batch. Input otherwise follows the existing normalized BGR 48×640 preprocessing.

| Endpoint | Purpose |
|---|---|
| `POST /v1/recognize` | One encoded crop; PNG/JPEG/WebP and other Pillow-supported formats |
| `POST /v1/recognize/batch` | 1–32 base64 crops, results in input order |
| `GET /v1/models` | Model identity, weight version and hardware mode |
| `GET /v1/stats` | Pending/completed/failed/rejected image jobs and worker initialization information |
| `GET /metrics` | Prometheus text counters and pending gauge |
| `GET /healthz`, `/readyz` | Process liveness and GPU readiness |

## Limits and recovery

Defaults: two GPU workers, ten in-flight crops and 32 admitted image jobs including queued work. These are shared across callers, not multiplied per request. Uvicorn runs one process; ten HTTP callers do not create ten models. Maximum image size is 5 MiB/20 megapixels; a request body is capped at 16 MiB. Images are not saved or logged.

| Setting | Default | Meaning |
|---|---:|---|
| `OCR_GPU_WORKERS` | 2 | Native GPU workers, range 1–4 |
| `OCR_MAX_INFLIGHT` | 10 | In-flight jobs, workers through 32 |
| `OCR_MAX_PENDING` | 32 | Total admitted jobs, in-flight through 128 |
| `OCR_REQUEST_TIMEOUT` | 30 | Queue + preprocessing + inference deadline, 1–300 seconds |
| `OCR_API_KEY` | empty | Optional bearer authentication for inference, model information and metrics |

The sample Compose file sets worker/queue defaults explicitly; edit those environment values to tune them. Full queues return **429 + Retry-After**, invalid images return **422**, excessive bodies/images **413**, unavailable GPU **503**, and inference deadlines **504**. Batch queue admission is all-or-none. Work already sent to GPU retains its capacity until complete, even when the client disconnects. An unexpected worker failure exits the API; Docker's restart policy reloads the service. This is application-process recovery, not recovery from a kernel hang or loss of power.

## Build from open source

On an ARM64 Linux build host:

```bash
docker build -t snapdragon855-ocr-api:0.3.0 .
```

On another architecture, use Buildx with `--platform linux/arm64 --load`. The default Dockerfile builds pinned ncnn + glslang and the native worker from source. A cache/cross-build path avoids rebuilding a verified ncnn dependency:

```bash
cmake --build build --target pnnx-gpu-worker -j2
docker build -f docker/runtime.Dockerfile -t snapdragon855-ocr-api:0.3.0 .
```

First configure `build/` using the [GPU guide](gpu-native.md), with an aarch64 toolchain for cross compilation. Build the runtime image for the same target architecture as the executable. The published ARM64 image uses this verified cache/cross-build path; CI also compiles the default source Dockerfile on amd64. The latter is a build check, not an amd64 GPU qualification.

## Verification and licensing

<!-- api-measurement -->
Measured: **6.250 crops/s**, 10 concurrent clients, client p50 **1599 ms**, p95 **1603 ms**. These latency percentiles include the queue under ten-way load, not isolated single-request GPU time. Worker-process termination/restart and a real post-recovery GPU inference passed.
<!-- /api-measurement -->

[API benchmark data](../benchmarks/api-v0.3.json) records real HTTP requests over the physical USB link, with image decode/preprocessing included. The 310-crop recognition outputs and hashes match v0.2, including 257 correct lines. The corpus has engineering exposure and is not an unseen accuracy test. Native CLI and API measurements were separate runs; small throughput differences are noise, not an API speed-up claim.

See [licensing](licensing.md) and `/app/licenses` in the image. The GPU worker, ncnn and glslang are open source; Debian/Python packages retain their licenses. Models are mounted separately from the unchanged v0.2 open asset. Qualcomm SDK/runtime/firmware and compiled HTA caches are not bundled. API implementation follows FastAPI's [lifespan model](https://fastapi.tiangolo.com/advanced/events/); device and read-only mounts use standard [Docker options](https://docs.docker.com/engine/containers/run/).
