#!/bin/sh
set -eu
python /app/scripts/verify_gpu_models.py "${OCR_MODEL_DIR:-/models}" --runtime-only
exec python -m uvicorn service.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 1 --limit-concurrency 64 --timeout-keep-alive 15 --timeout-graceful-shutdown 10 --no-access-log
