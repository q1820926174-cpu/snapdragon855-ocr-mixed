# Controlled PNNX GPU comparison

[Home](../README.md) · **English** | [简体中文](pnnx-benchmarks.zh-CN.md)

Medians of two complete 310-crop runs; second-round order reversed.

| Configuration | Correct / 310 | Crops/s ↑ | System W | J/crop ↓ |
|---|---:|---:|---:|---:|
| Previous pure GPU | 256 | 4.591 | 4.344 | 0.946 |
| **PNNX native GPU** | 257 | 6.243 | 4.233 | 0.678 |
| Mixed `quality` | 255 | 2.463 | 3.697 | 1.502 |
| Mixed `fast` | 252 | 5.243 | 3.727 | 0.711 |

Per-run measurements and ranges are in [pnnx-controlled.json](../benchmarks/pnnx-controlled.json). Power CSV paths are relative to `benchmarks/`; time starts with admission of the first crop. Integrating each CSV by the trapezoid rule and dividing by 310 reproduces J/crop.

| Profile | Round | Cooldown °C | First sample °C | Maximum °C | Seconds | Crops/s | J/crop |
|---|---:|---:|---:|---:|---:|---:|---:|
| Previous pure GPU | 1 | 35.5 | 35.5 | 37.1 | 67.507 | 4.592 | 0.942 |
| **PNNX native GPU** | 1 | 35.5 | 35.5 | 36.5 | 49.673 | 6.241 | 0.641 |
| Mixed `quality` | 1 | 35.5 | 35.5 | 37.7 | 128.451 | 2.413 | 1.533 |
| Mixed `fast` | 1 | 35.5 | 35.5 | 36.5 | 59.491 | 5.211 | 0.734 |
| Previous pure GPU | 2 | 35.5 | 35.5 | 37.0 | 67.543 | 4.590 | 0.951 |
| **PNNX native GPU** | 2 | 35.4 | 35.4 | 36.5 | 49.637 | 6.245 | 0.716 |
| Mixed `quality` | 2 | 33.9 | 33.9 | 36.3 | 123.385 | 2.512 | 1.471 |
| Mixed `fast` | 2 | 35.5 | 35.5 | 36.5 | 58.772 | 5.275 | 0.688 |

The common device lock is held during cooldown, initialization and inference. Battery temperature must read ≤35.5°C in three checks five seconds apart; the screen stays DPMS-off. Two GPU workers or four mixed workers share ten bounded in-flight requests, batch size one. No clocks, thermal or charging policy are changed.

Power is sampled at approximately 2 Hz from ISYS × VSYS. Samples with VSYS age ≥6 seconds are excluded, boundaries interpolated, and energy integrated over first admitted crop to last completed crop. Initialization and six-second idle intervals are excluded. This is a whole-system rail estimate including the desktop, without idle subtraction. Battery temperature is only a thermal proxy; it does not establish identical GPU junction temperature. Two rounds are not confidence intervals.

Native GPU energy varies across repeats, and its range overlaps mixed fast. The small median gap is not evidence of a repeatable energy lead over fast. One quality repetition began at 33.9°C; the shared cooling rule is an upper bound, not an exact match of starting temperatures. The four pure-GPU repetitions began at 35.4–35.5°C.

## Reliability and precision

An initial `mixedquality` run stopped after 192 crops with SNPE TF8 error 910 / DSP runtime error -6. The guard restored the normal boot image and the NPU firmware power state. The incomplete attempt is excluded; two subsequent complete runs are shown. This remains a mixed-runtime reliability limitation, not a PNNX GPU failure.

All repeated outputs match within each profile. Native GPU retains all old-GPU-correct and CPU-correct lines, with 257/310 total. Mixed fast has five new CPU-correct losses, so its speed/energy does not represent equal accuracy. Auxiliary 100 has 87 correct for both GPU versions, but native GPU adds one character edit. Read the [model card](model-card.md) and [precision data](../benchmarks/pnnx-precision.json); these corpora have engineering exposure.

The [public rebuild](../benchmarks/pnnx-recheck.json) checks all 310 outputs using the shipped source and run script. FP32 source-conversion parity is a separate check from FP16 phone execution. Earlier exploratory runs and historical CPU data are not folded into these medians. Global subgroup64 and tanh-GELU experiments were not selected. A diagnostic layer profile used host readback to expose GPU timestamps and is not used for production throughput claims.
