#!/bin/sh
set -eu
apt-get update
apt-get install -y --no-install-recommends ca-certificates python3 python3-venv libvulkan1 libgomp1 libstdc++6 libzstd1 libxcb-dri3-0 libx11-xcb1 libxcb-present0 libxcb-xfixes0 libxcb-sync1 libxcb-randr0 libxcb-shm0 libxshmfence1
rm -rf /var/lib/apt/lists/*
python3 -m venv /opt/venv
/opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements-api.txt
useradd --uid 10001 --create-home --shell /usr/sbin/nologin ocr
