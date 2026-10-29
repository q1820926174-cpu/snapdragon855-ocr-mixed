# Redistribution boundary

[Home](../README.md) · **English** | [简体中文](licensing.zh-CN.md)

The open release consists of original integration code, the ncnn patch, and model-owned graph/weight/quantization tensors. Original code is Apache-2.0. The PaddleOCR project and the official [PP-OCRv6 medium recognition model card](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/blob/main/README.md) declare Apache-2.0. Derived model weights and dictionary retain attribution and change notices in `NOTICE`. The ncnn patch retains upstream BSD-3-Clause attribution.

The installed Qualcomm SNPE 1.59 SDK's license was inspected locally during release review. Its software, tools and documentation have separate restricted distribution terms. This review did **not** establish that the SDK's headers, libraries or generated HTA compiler-cache blobs may be relicensed under Apache-2.0. The public release therefore does not upload them. This is a conservative release boundary, not a claim that every separately licensed distribution of a DLC is prohibited.

Users obtain their own authorized SDK, FastRPC/runtime and firmware, and generate their HTA containers locally. Calling SDK APIs from our integration code does not make the SDK itself open source. The build recipes reference it as an external dependency and do not download unofficial mirrors or ask for credentials.

The separate **v0.2 PNNX GPU package** does not depend on those Qualcomm components. It includes the original open ONNX source and derived ncnn graphs under the PaddleOCR model's Apache-2.0 attribution. PNNX is implemented in Tencent/ncnn/tools/pnnx under the ncnn license; converter binaries and their dependencies are not bundled. The original v0.1 release asset is unchanged.

No calibration/test image files, raw image tensors, per-image transcripts, SDK license PDF, SDK samples, private SSH files, Wi-Fi information, firmware, boot images, prebuilt ELF executors or third-party runtime binaries are published. The original private all-in-one device bundle is **not** the GitHub release asset. The GitHub asset is a separately allowlisted open-model package with reproducible local conversion.
