# Cached/cross build path: build/pnnx-gpu-worker must be a Linux target binary
# rebuilt from this repository with the documented pinned ncnn dependency.
FROM debian:bookworm-slim
LABEL org.opencontainers.image.source="https://github.com/q1820926174-cpu/snapdragon855-ocr-mixed" org.opencontainers.image.version="0.3.0"
COPY requirements-api.txt /tmp/requirements-api.txt
COPY docker/install-runtime.sh /tmp/install-runtime.sh
RUN sh /tmp/install-runtime.sh
COPY build/pnnx-gpu-worker /usr/local/bin/pnnx-gpu-worker
WORKDIR /app
COPY service/ /app/service/
COPY scripts/prepare_inputs.py scripts/verify_models.py scripts/verify_gpu_models.py /app/scripts/
COPY docker/entrypoint.sh /app/entrypoint.sh
COPY LICENSE NOTICE /app/
COPY licenses/ /app/licenses/
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OCR_MODEL_DIR=/models OCR_GPU_WORKERS=2 OCR_MAX_INFLIGHT=10 OCR_MAX_PENDING=32 OCR_REQUEST_TIMEOUT=30 XDG_CACHE_HOME=/tmp/ocr-cache VK_DRIVER_FILES=/opt/scv41-mesa-kgsl/share/vulkan/icd.d/freedreno_icd.armv8-a.json LD_LIBRARY_PATH=/opt/scv41-mesa-kgsl/lib:/opt/scv41-mesa-kgsl/lib/aarch64-linux-gnu
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz',timeout=3)"]
ENTRYPOINT ["sh", "/app/entrypoint.sh"]
