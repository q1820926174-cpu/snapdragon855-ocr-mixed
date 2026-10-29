# Benchmarks and power interpretation

[Home](../README.md) · **English** | [简体中文](benchmarks.zh-CN.md)

Machine-readable results are in [results.json](../benchmarks/results.json).

## Scope

- The corpus contains 310 fixed development regression text crops at 48×640 input size.
- Throughput and energy figures count **recognition crops**, not whole photos or pages.
- Model initialization is excluded.
- Mixed profiles use four workers in one process sharing 12 or 16 NPU contexts, with ten bounded in-flight requests. They do not launch ten independent SNPE processes.

## Results

| Profile | Correct/310 | New errors vs CPU-correct crops | Crops/s | System W | J/crop |
|---|---:|---:|---:|---:|---:|
| CPU | 255 | 0 | 2.887 | 5.052 | 1.750 |
| GPU | 256 | 0 | 4.584 | 4.544 | 0.991 |
| quality | 255 | 0 | 2.428 | 3.930 | 1.618 |
| fast | 252 | 5 | 5.305 | 3.907 | 0.736 |

**Pure GPU remains the everyday recommendation on the tested phone.** Quality retained CPU-correct samples but was slower and used more energy per crop than pure GPU. Fast traded accuracy for higher throughput and lower energy use.

## Why lower power does not always mean much lower energy

**Power measures how quickly energy is consumed; energy per crop also depends on how long each crop takes.** Quality uses about 22% less system power than CPU here, but its lower throughput leaves only about 7.5% lower energy per crop. Fast is about 58% lower energy per crop, with accuracy regressions. The earlier pure-GPU result is more energy-efficient than quality. These are approximate comparisons, not a claim of equal-quality 58% NPU energy savings.

## Measurement method

Power is estimated from the qualified Samsung fuel gauge: `ISYS_mA × VSYS_mV / 1e6` watts, sampled around the actual inference interval. The driver's `power_now` field reports current in this configuration; interpreting it as microwatts would be wrong. This measures the **system rail including the desktop**, not isolated NPU/GPU die power, external wall power, or an incremental accelerator power measurement.

## Limits and validation

Baseline batches were run at different times and temperatures. Battery maxima were approximately CPU 37.6°C, pure GPU from its separate run, quality 39.7°C and fast 37.2°C. Background load, charging state and temperature can change the estimates. A 7.5% difference needs controlled same-temperature repetition before drawing a strong energy-efficiency conclusion.

The corpus was used during development choices, so it is not an unseen final test set. Results on these crops do not establish lossless accuracy on new images.

The guard restores the normal boot state after device tests; the runtime also restores the prior NPU firmware power state. Physical acceptance was run on the host/established WSL with the real SCV41. CI only checks source/input contracts. The original private per-image evidence and device identifiers are not uploaded.
