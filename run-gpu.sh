#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ];then echo 'Usage: bash run-gpu.sh input.tsv output.jsonl [1..310]' >&2;exit 2;fi
limit=${3:-310}
if ! [[ "$limit" =~ ^[1-9][0-9]{0,2}$ ]] || [ "$limit" -gt 310 ];then echo 'Limit must be 1..310' >&2;exit 2;fi
input=$(readlink -e -- "$1")
output=$(readlink -m -- "$2")
if [ "$input" = "$output" ];then echo 'Input list and output must differ' >&2;exit 2;fi
assets=${SCV41_GPU_ASSETS:-"$repo/models/gpu-native"}
binary=${SCV41_GPU_BINARY:-"$repo/build/pnnx-gpu-runner"}
test -x "$binary"
test -r "$assets/cnn/model.param"
test -r "$assets/tail/model.param"
export SCV41_PIPE_ENGINE=gpu SCV41_PIPE_BASE="$assets" SCV41_PIPE_INPUT_LIST="$input" SCV41_PIPE_OUTPUT="$output" SCV41_PIPE_LIMIT="$limit"
export SCV41_PIPE_WORKERS=2 SCV41_PIPE_CONCURRENCY=10 SCV41_PIPE_CPU_THREADS=1 SCV41_PIPE_OFFSET=0
export SCV41_PNNX_SPLIT=1 SCV41_PNNX_CNN_FP32=0 SCV41_PNNX_CNN_HALF_ARITH=1
export SCV41_NCNN_SAFE_GEMM=1 SCV41_NCNN_FP32_SLM64=1 SCV41_NCNN_STRICT_GPU=1
export VK_DRIVER_FILES=${VK_DRIVER_FILES:-/opt/scv41-mesa-kgsl/share/vulkan/icd.d/freedreno_icd.armv8-a.json}
export LD_LIBRARY_PATH="/opt/scv41-mesa-kgsl/lib/aarch64-linux-gnu:/usr/local/lib:${LD_LIBRARY_PATH:-}"
unset SCV41_PNNX_SUBGROUP64 SCV41_NCNN_NODE_SYNC
exec "$binary"
