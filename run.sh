#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ];then
 echo 'Usage: sudo bash run.sh quality|fast input.tsv output.jsonl [1..310]' >&2;exit 2
fi
profile=$1
case "$profile" in quality|fast) ;; *) echo 'Unknown profile' >&2;exit 2;;esac
limit=${4:-310}
if ! [[ "$limit" =~ ^[1-9][0-9]{0,2}$ ]] || [ "$limit" -gt 310 ];then echo 'Limit must be 1..310' >&2;exit 2;fi
input=$(readlink -e -- "$2")
output=$(readlink -m -- "$3")
if [ "$input" = "$output" ];then echo 'Input list and output must differ' >&2;exit 2;fi
assets=${SCV41_MODEL_ASSETS:-"$repo/models"}
bin=${SCV41_MODEL_BIN:-"$repo/build"}
export SCV41_PIPE_INPUT_LIST="$input" SCV41_PIPE_OUTPUT="$output" SCV41_PIPE_LIMIT="$limit"
export SCV41_PIPE_ENGINE=mixed SCV41_PIPE_BASE="$assets/gpu" SCV41_PIPE_WORKERS=4 SCV41_PIPE_CONCURRENCY=10 SCV41_PIPE_BATCH=1 SCV41_PIPE_OFFSET=0 SCV41_PIPE_CPU_THREADS=1
export SCV41_NPU_TEST_TIMEOUT=170 SCV41_NCNN_SAFE_GEMM=1 SCV41_NCNN_STRICT_GPU=1 SCV41_NCNN_FP32_SLM64=1 SCV41_GPU_CTC_MODE=2
export SCV41_SNPE_NATIVE_LIBRARY_DIR=/opt/scv41-mesa-kgsl/lib/aarch64-linux-gnu:/usr/local/lib
export VK_DRIVER_FILES=/opt/scv41-mesa-kgsl/share/vulkan/icd.d/freedreno_icd.armv8-a.json
unset SCV41_GPU_HALF_CNN SCV41_NCNN_NODE_SYNC SCV41_855_CORRECTIONS SCV41_855_CALIBRATE SCV41_PIPE_FALLBACK_THRESHOLD SCV41_PIPE_DIAG
export SCV41_855_MODEL_ROOT="$assets/npu"
if [ "$profile" = quality ];then
 export SCV41_NPU_TEST_BINARY="$bin/mixed-refined-fp32-bench" SCV41_855_NPU_IDS=74,80,83,89,98,104,107,113,122,128,131,137 SCV41_855_REFINE=1
 unset SCV41_855_PACKED_BRIDGE
else
 export SCV41_NPU_TEST_BINARY="$bin/mixed-packed-bench" SCV41_855_NPU_IDS=all SCV41_855_PACKED_BRIDGE=1
 unset SCV41_855_REFINE
fi
test -x "$SCV41_NPU_TEST_BINARY"
test -r "$VK_DRIVER_FILES"
test -c /dev/msm_npu
test -r "$assets/npu/n_74/int8-hta.dlc"
exec python3 "$repo/runtime/npu-manager.py"
