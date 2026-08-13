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
CADENCE_STAGE("label", stream)    // GPU work, one event, chained to the previous stage
CADENCE_SCOPE("label")            // CPU span, steady_clock
CADENCE_FLUSH()                   // consume pending records; synchronizes
CADENCE_REPORT()                  // flush and write the CSV

cadence::Configure(cfg)           // warmup, output path, NVTX, sampling, enable
cadence::Snapshot()               // vector<Stats>, for printing or asserting on
cadence::Reset()                  // drop statistics, restart warmup
```

`-DCADENCE_DISABLE` compiles every macro to nothing, the way `-DNVTX_DISABLE` does for NVTX.

`CADENCE_WARMUP`, `CADENCE_OUTPUT`, `CADENCE_NVTX`, `CADENCE_SAMPLE` and `CADENCE_ENABLE` override the config struct at runtime.

## How it measures

Timing uses CUDA events on the work's stream, not wall clock. Wall clock around an async launch measures launch overhead (5-15 µs), not the kernel.

`Flush()` is the only synchronization point. Event pairs are buffered and resolved there, so instrumentation never serializes the pipeline it is measuring. Call it at a loop boundary.

The first N iterations per label are discarded (default 3). They pay for context creation, JIT and cuBLAS/cuDNN autotuning.

Every scope also pushes an NVTX range, so the same instrumentation shows up in Nsight Systems.

The CSV header records device, compute capability, clock ceilings and warmup count. cadence does not lock clocks; use `nvidia-smi -lgc <mhz>` before comparing runs.

## Overhead

Best difference of 7 × 50,000 scopes, RTX A4000:

```
CADENCE_KERNEL            3.9 µs/scope
CADENCE_STAGE             2.5 µs/scope   (64% of CADENCE_KERNEL)
CADENCE_KERNEL, 1 in 10   0.5 µs/scope   (12% of CADENCE_KERNEL)
CADENCE_SCOPE             0.32 µs/scope
NVTX passthrough          ~20 ns/scope
```

Almost all of `CADENCE_KERNEL` is two `cudaEventRecord` calls at ~1.4 µs each. That cost belongs to the CUDA API, not to cadence: the same call with `cudaEventDisableTiming` runs in 153 ns, because reading a timestamp is the expensive part, and it is flat regardless of how many records you batch or how deep the launch queue is. Nothing cadence does on top of it — the pool, the record, the flush — accounts for more than a few percent.

So the only real lever is recording fewer events, which is what the other two rows are.

**`CADENCE_STAGE` records one event per scope instead of two**, using the previous stage's stop event as its own start. The price is a different measurement: a stage is timed from the end of the previous stage on that stream, so anything you enqueue between two stages is charged to the later one. Use it when every launch on the stream is inside some stage. Use `CADENCE_KERNEL` when it is not, or when one number has to stand on its own.

**`sampleEvery` measures one iteration in N.** It divides the cost rather than shaving it, and it spends coverage to do so: a once-a-minute outlier is now something you may simply miss. The CSV header records the rate so the counts stay readable.

Wrap stages, not individual small kernels: below ~50 µs of GPU work per scope, the instrumentation is a visible fraction of the result.

`CADENCE_SCOPE` is two `steady_clock` reads. Both host figures are inflated here — the test machine is a Xen VM where `steady_clock::now()` costs ~151 ns against ~20-30 ns on a `tsc` clocksource. Measure on your own hardware; `benchmarks/overhead.cu` is the program that produced the table above.

## Building the tests

```sh
cmake -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The host tests, the disabled-build test and the self-contained-header test need no GPU and no CUDA toolkit. Device tests build when `nvcc` is found and skip themselves when no device answers.

## License

MIT
