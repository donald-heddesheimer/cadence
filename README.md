# cadence

Time the stages of a CUDA loop from inside your own process. Header-only, C++17.

[![ci](https://github.com/donald-heddesheimer/cadence/actions/workflows/ci.yml/badge.svg)](https://github.com/donald-heddesheimer/cadence/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![standard](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![header-only](https://img.shields.io/badge/header--only-yes-brightgreen.svg)](include/cadence)

Wrap the stages you care about, run your application normally, and it prints a
latency distribution when it exits. No profiler to launch, no process to attach,
no capture file to open. The instrumentation compiles into your binary, so you
can leave it in and look at every run instead of only the ones you thought to
profile.

## Quick start

```cpp
#include <cadence/cadence.h>

while (running) {
  CADENCE_SCOPE("iteration");                                 // CPU span

  { CADENCE_KERNEL("saxpy", stream); Saxpy<<<...>>>(...); }   // GPU span
  { CADENCE_KERNEL("scale", stream); Scale<<<...>>>(...); }

  cudaStreamSynchronize(stream);
  CADENCE_FLUSH();          // once per iteration, never between scopes
}
CADENCE_REPORT();
```

![cadence report: a table of per-label latency distributions with a histogram column, a deadline verdict reading MISSED on 185 of 190 iterations, and a breakdown of the three slowest iterations by stage](docs/report.svg)

Reading it:

- **Two rows per label.** `device` is GPU execution measured with CUDA events;
  `host` is CPU time spent issuing the work. When they converge, the stage is
  launch-bound rather than compute-bound.
- **The distribution column** is what a mean hides. A lone mark far right is a
  stall, and no column of averages would tell you it existed.
- **The deadline line** counts misses instead of averaging overshoot, because a
  loop that blows its budget once in fifty has a healthy mean and a real problem.
- **The slowest iterations** are kept whole, so you see which passes missed and
  where their time went, not just that some did.

## How it stays out of the way

Timing GPU work usually means synchronizing to read a stopwatch, which changes
the thing being timed. cadence records events and walks away; the numbers are
resolved later, at a boundary where you were already going to synchronize.

Buffers are thread-local, so threads do not contend; statistics are folded in
with Welford's method and a sample reservoir, so a loop running for a week costs
the same memory as one running for a minute.

## Timelines

Set `tracePath` and the same retained iterations are written as Chrome Trace
Event JSON, which [ui.perfetto.dev](https://ui.perfetto.dev) opens directly:

```cpp
cfg.tracePath = "worst.json";   // or: CADENCE_TRACE=worst.json ./app
```

![One iteration on the Perfetto timeline. On the host lane both kernel launches have returned within the first 10µs; on the device lane saxpy runs until 41µs and scale does not begin until 43µs, leaving a visible gap between a launch and the work it queued](docs/timeline.png)

One lane for the CPU, one per CUDA stream, on a shared clock. Above, `scale` was
queued at 10µs and did not start on the GPU until 43µs — a gap you would never
infer from two columns of durations. Only the worst iterations are exported; a
trace of a whole run is enormous and nobody scrolls a million spans looking for
the bad one.

## Where it fits

| | Answers | How you run it | Live loop? |
|---|---|---|---|
| **cadence** | which stage got slow, and how much it varies | `#include` it; ships in your binary | yes, by design |
| Nsight Systems | what the whole system did over a few seconds | launch under `nsys`, open the capture | one-off investigation |
| Nsight Compute | why one kernel is slow, to hardware counters | `ncu`, which replays each kernel | no, orders of magnitude slower |
| nvbench | how a kernel scales across a parameter sweep | a separate benchmark binary | no, measures kernels not your app |
| Hand-rolled `cudaEvent` | whatever you wired up | you write it | usually wrong; the obvious version synchronizes per kernel |

cadence complements the Nsight tools. Use it to find *when and where* you got
slow, then `nsys` or `ncu` for *why*. Every scope also emits an NVTX range, so
the same instrumentation shows up on the Nsight timeline for free.

## Features

- **Distributions, not averages.** mean, min, p50, p95, max, stddev, jitter.
- **Exact where it counts.** Past the retention cap, count, mean, stddev, min,
  max and the deadline verdict stay exact; only percentiles and the histogram
  become estimates, drawn from a uniform sample of the whole run rather than a
  recent window.
- **Colour when a terminal is watching.** Auto-detected, and `NO_COLOR` is
  honoured; a redirected run and the `outputPath` copy stay clean text.
- **CPU and GPU through one API.** Host timers compile in translation units with
  no CUDA in them.
- **Compiles to nothing.** `-DCADENCE_DISABLE` removes every macro without
  leaving a runtime branch behind.
- **Provenance in the output.** Device, clock ceilings, warmup, sampling rate and
  any dropped records, so a number traces back to the run that produced it.
- **Header-only.** No third-party dependencies, MIT licensed.

## Install

```cmake
add_subdirectory(cadence)                              # vendored
target_link_libraries(my_app PRIVATE cadence::cadence)
```

```cmake
include(FetchContent)                                  # fetched
FetchContent_Declare(cadence
  GIT_REPOSITORY https://github.com/donald-heddesheimer/cadence.git
  GIT_TAG        v0.2.0)
FetchContent_MakeAvailable(cadence)
target_link_libraries(my_app PRIVATE cadence::cadence)
```

Or copy `include/cadence` onto your include path and skip CMake entirely.

## API

| Macro | Measures | ns/scope |
|---|---|---:|
| `CADENCE_KERNEL("label", stream)` | GPU span, paired CUDA events | 3390 |
| `CADENCE_STAGE("label", stream)` | GPU span, one event, chained to the previous stage | 2410 |
| `CADENCE_SCOPE("label")` | CPU span, `steady_clock` | 330 |
| `CADENCE_FLUSH()` | resolves pending records; synchronizes | once per loop |
| `CADENCE_REPORT()` | flush, then print the report | once per run |

`stream` is optional and defaults to the default stream.

```cpp
cadence::Config cfg;
cfg.warmupIterations = 10;      // discard context creation, JIT, cuBLAS autotuning
cfg.budgetMs = 0.080;           // deadline for one stage; 0 disables the check
cfg.budgetLabel = "";           // which stage; empty picks the loop span
cfg.numWorstIterations = 3;     // slowest iterations kept whole; 0 turns it off
cfg.tracePath = "worst.json";   // Perfetto-openable timeline of those iterations
cfg.sampleEvery = 1;            // measure one iteration in N
cfg.maxSamplesPerLabel = 32768; // retained per row; 0 keeps every observation
cfg.outputPath = "run.txt";     // also write the report here
cfg.reportStream = &std::cerr;  // where it prints; nullptr suppresses printing
cfg.unicodeOutput = false;      // ASCII table for terminals that mangle UTF-8
cfg.colorOutput = cadence::ColorMode::Auto;  // Auto colours a terminal and nothing else
cadence::Configure(cfg);

std::vector<cadence::Stats> rows = cadence::Snapshot();   // assert on these in tests
std::vector<cadence::TraceIteration> worst = cadence::WorstIterations();
```

Every field has an environment override (`CADENCE_WARMUP`, `CADENCE_BUDGET_MS`,
`CADENCE_TRACE`, `CADENCE_ENABLE`, ...), so a deployed binary can be re-pointed
without a rebuild. Set the budget before the loop starts: misses are counted as
observations arrive, which is what keeps the verdict exact on a long run.

## Overhead

Best difference over 7 x 50,000 scopes on an RTX A4000. Nearly all of
`CADENCE_KERNEL` is what the CUDA API charges for a timestamped event, not
anything cadence does; the same call with `cudaEventDisableTiming` costs 153 ns.
About 200 ns per device scope is the CUDA-graph-capture guard and the device
query that keeps events paired with the GPU that created them, neither of which
is optional because the failures they prevent are silent.

| | ns/scope | |
|---|---:|---|
| `CADENCE_KERNEL` | 3390 | two `cudaEventRecord` calls |
| `CADENCE_STAGE` | 2410 | one event, chained |
| `CADENCE_KERNEL`, 1 in 10 | 367 | sampled |
| `CADENCE_SCOPE` | 330 | two `steady_clock` reads |
| NVTX passthrough | ~10 | with no tool attached |

Wrap stages, not individual small kernels. Below roughly 50 µs of GPU work per
scope the instrumentation becomes a visible fraction of the result.
[docs/overhead.md](docs/overhead.md) has the measurements behind this.

## Build and test

```sh
cmake -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
```

Host, disabled-build and self-contained-header tests need no GPU and no CUDA
toolkit. Device tests build when `nvcc` is found and skip themselves when no
device answers.

## What it does not do

Elapsed time only, through the CUDA runtime API: hardware counters are `ncu`'s
job, system-wide timelines are `nsys`'s.

Work captured into a CUDA graph is not measured. A `cudaEventRecord` issued into
a capturing stream is baked into the graph rather than executed, which makes the
event permanently unreadable and, if a flush lands before the capture closes,
invalidates the capture itself. Scopes on a capturing stream therefore record
nothing and the report says how many stood down. Wrap the graph launch instead.

Events are pooled per device, so a process driving several GPUs measures each
correctly, but the report lists labels rather than devices: two GPUs running the
same label share one row.

## License

MIT
