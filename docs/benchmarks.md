# Benchmarks and power interpretation

Machine-readable results are in [results.json](../benchmarks/results.json). All throughput and energy figures count **recognition crops**, not whole photos or pages. They exclude model initialization. Ten in-flight requests are served by four workers in one process sharing 12 or 16 NPU contexts; this is not ten independent SNPE processes.

| Profile | Correct/310 | New errors vs CPU-correct crops | Crops/s | System W | J/crop |
|---|---:|---:|---:|---:|---:|
| CPU | 255 | 0 | 2.887 | 5.052 | 1.750 |
| GPU | 256 | 0 | 4.584 | 4.544 | 0.991 |
| quality | 255 | 0 | 2.428 | 3.930 | 1.618 |
| fast | 252 | 5 | 5.305 | 3.907 | 0.736 |

**Lower watts does not imply proportionally lower joules per crop.** Quality uses about 22% less system power than CPU here, but its lower throughput leaves only about 7.5% lower energy per crop. Fast is about 58% lower energy per crop, with accuracy regressions. The earlier pure-GPU result is more energy-efficient than quality. These are approximate comparisons, not a claim of equal-quality 58% NPU energy savings.

Power is estimated from the qualified Samsung fuel gauge: `ISYS_mA × VSYS_mV / 1e6` watts, sampled around the actual inference interval. The driver's `power_now` field reports current in this configuration; interpreting it as microwatts would be wrong. This measures the **system rail including the desktop**, not isolated NPU/GPU die power, external wall power, or an incremental accelerator power measurement.

Baseline batches were run at different times and temperatures. Battery maxima were approximately CPU 37.6°C, pure GPU from its separate run, quality 39.7°C and fast 37.2°C. Background load, charging state and temperature can change the estimates. A 7.5% difference needs controlled same-temperature repetition before drawing a strong energy-efficiency conclusion.

The guard restores the normal boot state after device tests; the runtime also restores the prior NPU firmware power state. Physical acceptance was run on the host/established WSL with the real SCV41. CI only checks source/input contracts. The original private per-image evidence and device identifiers are not uploaded.
