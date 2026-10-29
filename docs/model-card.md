# Model card

[Home](../README.md) · **English** | [简体中文](model-card.zh-CN.md)

- Upstream: [PaddlePaddle PP-OCRv6_medium_rec](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx), Apache-2.0.
- Target: native Debian 12 on Samsung SCV41, Snapdragon 855/SM8150 HTA and Adreno 640.
- Task: recognition of already cropped text, not detection or full-page OCR.
- Input: normalized BGR NHWC float32 `[1,48,640,3]`, right-padded with normalized zero. Output: greedy CTC text, 80 steps and 18,710 symbols.
- Original model export hashes, graph/layer dimensions, fixed-point encodings and asset hashes are in `models/model-spec.json` and `models/SHA256SUMS.json` after downloading the open asset.

## Mixed precision

The **v0.2 native GPU path** is a separate export from the original ONNX, using PNNX 20261020. It uses FP16 CNN storage/arithmetic with an FP32 sequence/classifier tail, no INT8 model, no NPU contexts and no SNPE dependency. Its assets and hashes are under `models/gpu-native/`; see the [GPU guide](gpu-native.md). The original mixed profiles below remain available.

Large 1×1 convolutions are centered and channel-balanced. The NPU convolutions are bias-free W8A8. GPU FP32 restores inverse output scales, original-domain bias, and clipped weight outliers using CSR sparse dot products. Nonlinearities, normalization, attention and classifier remain on GPU.

`quality` routes the first 12 selected convolutions to the NPU and keeps all remaining GPU storage/arithmetic FP32. It executes a second W8A8 pass on the scaled input quantization residual and combines results on GPU. The model uses the same NPU weight grid twice; it is not W8A16 hardware support. `fast` uses all 16 selected convolutions with one pass and fuses TF8/NHWC conversion with GPU FP16 pack4 storage.

## Calibration and selection

The shipped PTQ model used 20 independent calibration crops (10 English, 10 Chinese). Raw SHA256 checks excluded overlap with the 310-crop regression. Additional experiments used 130 independent activation-calibration crops and an 80-training/20-heldout teacher-distillation split. QAT, direct wider min/max and weighted clipping candidates were tested and rejected because measured regression quality was worse. The public default does not contain those learned or recalibrated weights.

The frozen W8 grid in each public `model.onnx` is stored as dequantized FP32 constants for the legacy converter. Fixed encodings accompany it. Local SDK conversion is required for physical INT8 inference; opening those ONNX files in a generic runtime does not engage the NPU.

## Evaluation limits

For native GPU, the 310-crop regression improved from 256 to 257 correct while retaining all old-GPU and CPU-correct lines. A supplementary 100-crop calibration corpus retained all 87 old-GPU-correct lines. Character edit counts were 835/2447 (old GPU 836/2447) and 30/651 (old GPU 29/651), respectively. Thus the line-retention gate passed, but **not every accuracy metric improved**. The supplementary corpus also has prior calibration exposure. The FP32 source conversion check and FP16 device regression check are distinct; neither establishes bitwise or universal losslessness.

The 310 images were a **fixed development regression set used during engineering choices**. They are not an unseen final test set. Quality retained all CPU-correct samples on this set; it is not a guarantee of lossless quantization on arbitrary data. Fast had five new errors even though its total correct count fell by three: it gained other cases while losing five previously correct ones.

Calibration/test datasets and their transcriptions are excluded from redistribution. Some calibration image sources did not declare an image redistribution license. The open package contains model-owned parameters, not the source images or their labels. Use your own properly licensed corpus for independent validation.

## Intended use and constraints

This is an experimental development and measurement integration. It has not been qualified for safety-critical OCR, every Samsung/SM8150 variant, a mainline kernel, modern HTP SDKs, Android or arbitrary camera inputs. The phone's desktop and remote-access services are separate from this repository.
