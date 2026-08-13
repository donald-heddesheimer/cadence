# cadence

Header-only CUDA timing for applications that run a loop.

Wrap a scope, get per-label GPU and CPU timing as CSV. No GUI, no daemon, no external profiler to launch.

```cpp
#include <cadence/cadence.h>

for (int step = 0; step < steps; ++step) {
  { CADENCE_KERNEL("gemm", stream); Gemm<<<grid, block, 0, stream>>>(...); }
  CADENCE_FLUSH();     // once per iteration, not per kernel
}
CADENCE_REPORT();      // writes cadence.csv
```

```
label,scope,count,warmup_discarded,mean_ms,min_ms,p50_ms,p95_ms,max_ms,stddev_ms,jitter_ms
gemm,device,190,10,0.037306,0.035840,0.036992,0.037888,0.038912,0.000552,0.003072
gemm,host,190,10,0.005387,0.004919,0.005166,0.005583,0.035344,0.002195,0.030425
```

Each label gets two rows. `device` is GPU time from CUDA events; `host` is CPU time spent issuing the work. If they converge, you are launch-bound.

## Install

```cmake
add_subdirectory(cadence)
target_link_libraries(my_app PRIVATE cadence::cadence)
```

Or copy `include/cadence` onto your include path. Requires C++17. The host timers work in translation units with no CUDA.

## API

```cpp
CADENCE_KERNEL("label", stream)   // GPU work, paired CUDA events (stream optional)
CADENCE_SCOPE("label")            // CPU span, steady_clock
CADENCE_FLUSH()                   // consume pending records; synchronizes
CADENCE_REPORT()                  // flush and write the CSV

cadence::Configure(cfg)           // warmup, output path, NVTX, enable
cadence::Snapshot()               // vector<Stats>, for printing or asserting on
cadence::Reset()                  // drop statistics, restart warmup
```

`-DCADENCE_DISABLE` compiles every macro to nothing, the way `-DNVTX_DISABLE` does for NVTX.

`CADENCE_WARMUP`, `CADENCE_OUTPUT`, `CADENCE_NVTX` and `CADENCE_ENABLE` override the config struct at runtime.

## How it measures

Timing uses CUDA events on the work's stream, not wall clock. Wall clock around an async launch measures launch overhead (5-15 µs), not the kernel.

`Flush()` is the only synchronization point. Event pairs are buffered and resolved there, so instrumentation never serializes the pipeline it is measuring. Call it at a loop boundary.

The first N iterations per label are discarded (default 3). They pay for context creation, JIT and cuBLAS/cuDNN autotuning.

Every scope also pushes an NVTX range, so the same instrumentation shows up in Nsight Systems.

The CSV header records device, compute capability, clock ceilings and warmup count. cadence does not lock clocks; use `nvidia-smi -lgc <mhz>` before comparing runs.

## Overhead

Best of 5 x 50,000 iterations, RTX A4000:

```
CADENCE_KERNEL     ~3.8 µs/scope
CADENCE_SCOPE      ~0.56 µs/scope
NVTX passthrough   ~22 ns/scope
```

The kernel figure is two `cudaEventRecord` calls. Timing-enabled events force the driver to flush its launch queue, which is inherent to event timing rather than something cadence adds. Wrap stages, not individual small kernels: below ~50 µs of GPU work per scope, the instrumentation is a visible fraction of the result.

Both host figures are inflated here. The test machine is a Xen VM where `steady_clock::now()` costs ~169 ns against ~20-30 ns on a `tsc` clocksource, and each scope makes two calls. Measure on your own hardware.

## License

MIT
