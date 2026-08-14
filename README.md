# cadence

Time the stages of a CUDA loop from inside your own process. Header-only, C++17.

[![ci](https://github.com/donald-heddesheimer/cadence/actions/workflows/ci.yml/badge.svg)](https://github.com/donald-heddesheimer/cadence/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![standard](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![header-only](https://img.shields.io/badge/header--only-yes-brightgreen.svg)](include/cadence)

You wrap the stages you care about, run the application normally, and it prints a
latency distribution for each stage when it exits. There is no profiler to
launch, no process to attach, and no capture file to open afterwards. Since the
instrumentation compiles into your binary, you can leave it in and look at every
run rather than only the ones you thought to profile.

```cpp
#include <cadence/cadence.h>

while (running) {
  CADENCE_SCOPE("iteration");                                    // CPU span

  { CADENCE_KERNEL("saxpy", stream); Saxpy<<<...>>>(...); }      // GPU span
  { CADENCE_KERNEL("scale", stream); Scale<<<...>>>(...); }

  cudaStreamSynchronize(stream);
  CADENCE_FLUSH();          // once per iteration, never between scopes
}
CADENCE_REPORT();           // prints the report below
```

```
cadence report
  device    NVIDIA RTX A4000 (sm_86), sm clock <= 1560 MHz, mem clock <= 7001 MHz
  clocks    not locked by cadence; run `nvidia-smi -lgc 1560` before comparing runs
  warmup    10 iteration(s) discarded per label

  label      scope     n    mean     p50     p95     max  jitter  distribution
  ────────────────────────────────────────────────────────────────────────────
  iteration  host    190  70.3µs  68.6µs  70.0µs   367µs   302µs  █          ▂
  saxpy      device  190  37.3µs  36.9µs  37.9µs  41.0µs  5.12µs  ▂▂█▂▆▂▂▂   ▂
  saxpy      host    190  5.09µs  5.01µs  5.49µs  10.2µs  7.94µs  ▂  ▂█▂ ▂   ▂
  scale      device  190  24.2µs  24.6µs  24.6µs  25.6µs  3.07µs  ▂   ▇   █  ▂
  scale      host    190  5.14µs  5.06µs  5.50µs  8.04µs  3.31µs  ▇█▅▂▂      ▂

  device    61.5µs across 2 label(s)
  iteration  70.3µs, of which 87.5% is GPU work; 8.80µs is launch and synchronization

  elapsed time only; for occupancy or bandwidth use `ncu`.
```

Every label reports twice. `device` is GPU execution time measured with CUDA
events. `host` is the CPU time spent issuing the work. When the two rows for a
label converge, that stage is launch-bound rather than compute-bound.

The `distribution` column is why the report is worth reading rather than
grepping. `iteration` has a mean of 70.3 µs and a p50 of 68.6 µs, which looks
healthy, but one iteration took 367 µs. The histogram shows it as a lone mark
far to the right of the mass. That is the iteration that missed the deadline,
and no column of averages would have told you it existed.

## Deadlines

If your loop has to close at a fixed rate, give cadence the budget and it reports
how often you held it:

```cpp
cadence::Config cfg;
cfg.budgetMs = 0.100;   // this loop must close at 10 kHz
cadence::Configure(cfg);
```

```
  deadline  100µs on iteration (host)
  met       190/190 iterations inside budget (100.0%); worst 70.5µs at 71% of budget
            [███████████████████████████░░░░░░░░░░░░░]  p95 69.9µs at 70%
```

The verdict is a count of misses rather than an average overshoot, because a loop
that blows its deadline once in fifty has a healthy mean and a real problem. The
bar is drawn against p95, so the figure it shows is one you could plan against.

With no label named, the budget lands on the loop span: the one label that
recorded host time but never launched a kernel, which is the `CADENCE_SCOPE`
around your iteration. Set `cfg.budgetLabel` to hold a specific stage instead;
a named label with GPU work is held to its GPU time. If the choice is ambiguous
(two host-only labels, say), no row carries the budget rather than the wrong one.

## Where it fits

| | Answers | How you run it | Usable in a live loop? |
|---|---|---|---|
| **cadence** | which stage got slow, and how much it varies | `#include` it; ships in your binary | yes, that is what it is for |
| Nsight Systems | what the whole system did over a few seconds | launch under `nsys`, open the capture in a GUI | for one-off investigation |
| Nsight Compute | why one kernel is slow, down to hardware counters | `ncu`, which replays each kernel | no, the slowdown is orders of magnitude |
| nvbench | how a kernel scales across a parameter sweep | a separate benchmark executable | no, it measures kernels rather than your app |
| Hand-rolled `cudaEvent` | whatever you wired up | you write it | usually wrong, because the obvious version synchronizes per kernel |

cadence complements the Nsight tools rather than replacing them. Use it to find
when and where you got slow, then use `nsys` or `ncu` to find out why. Every
cadence scope also emits an NVTX range, so instrumentation you have already
added shows up on the Nsight timeline at no extra cost.

## Features

- **Two lines to instrument a stage.** A scope guard, and a flush at the loop boundary.
- **No synchronization on the hot path.** Event pairs are buffered and resolved at `CADENCE_FLUSH()`, which you put at a boundary where you were going to synchronize anyway. Instrumentation never serializes the pipeline it is measuring.
- **Distributions rather than averages.** mean, min, p50, p95, max, stddev, and jitter (max minus min) for every label. A loop with a deadline is decided by its tail.
- **Deadline reporting.** Give a stage a budget and the report says how many iterations held it, rather than leaving you to infer it from a percentile.
- **CPU and GPU through the same API.** The host timers compile in translation units with no CUDA in them, so one set of macros covers the whole pipeline.
- **Tunable overhead.** `CADENCE_STAGE` halves the number of events recorded, and `sampleEvery` measures one iteration in N. Both are opt-in, and [docs/overhead.md](docs/overhead.md) is explicit about what each one costs you in visibility.
- **Compiles to nothing.** `-DCADENCE_DISABLE` removes every macro without leaving a runtime branch behind, the same way `-DNVTX_DISABLE` works for NVTX.
- **Provenance in the output.** Every report opens with the device, compute capability, clock ceilings, warmup count, sampling rate, and any dropped records, so a number can be traced back to the run that produced it.
- **Header-only.** No third-party dependencies, MIT licensed.

## Install

```cmake
# vendored
add_subdirectory(cadence)
target_link_libraries(my_app PRIVATE cadence::cadence)
```

```cmake
# fetched
include(FetchContent)
FetchContent_Declare(cadence
  GIT_REPOSITORY https://github.com/donald-heddesheimer/cadence.git
  GIT_TAG        main)
FetchContent_MakeAvailable(cadence)
target_link_libraries(my_app PRIVATE cadence::cadence)
```

Or copy `include/cadence` onto your include path and skip CMake entirely.

## API

| Macro | Measures | Cost per scope |
|---|---|---|
| `CADENCE_KERNEL("label", stream)` | GPU span, paired CUDA events | 3.9 µs |
| `CADENCE_STAGE("label", stream)` | GPU span, one event, chained to the previous stage | 2.5 µs |
| `CADENCE_SCOPE("label")` | CPU span, `steady_clock` | 0.32 µs |
| `CADENCE_FLUSH()` | resolves pending records; synchronizes | once per loop |
| `CADENCE_REPORT()` | flush, then print the report | once per run |

The `stream` argument is optional and defaults to the default stream.

```cpp
cadence::Config cfg;
cfg.warmupIterations = 10;   // discard context creation, JIT, cuBLAS autotuning
cfg.outputPath = "run.txt";  // also write the report here; empty means print only
cfg.reportStream = &std::cerr;  // where it prints; nullptr suppresses printing
cfg.unicodeOutput = false;   // ASCII table for terminals that mangle UTF-8
cfg.sampleEvery = 1;         // measure one iteration in N
cfg.budgetMs = 0.100;        // deadline for one stage; 0 disables the check
cfg.budgetLabel = "";        // which stage; empty picks the loop span
cfg.nvtxEnabled = true;
cadence::Configure(cfg);

std::vector<cadence::Stats> rows = cadence::Snapshot();  // assert on these in tests
cadence::Reset();                                        // drop stats, restart warmup
```

`CADENCE_WARMUP`, `CADENCE_OUTPUT`, `CADENCE_NVTX`, `CADENCE_SAMPLE`,
`CADENCE_UNICODE`, `CADENCE_BUDGET_MS`, `CADENCE_BUDGET_LABEL` and
`CADENCE_ENABLE` override the config from the environment, so a deployed binary
can be re-pointed without a rebuild.

## How it measures

GPU time comes from CUDA events recorded on the work's own stream. A wall clock
wrapped around an asynchronous launch measures launch overhead, roughly 5 to 15
µs, and tells you nothing about the kernel.

`Flush()` is the only place cadence synchronizes. Records are buffered until
then, so put it at a loop or frame boundary and never between two scopes you
intend to compare.

The first N iterations of each label are discarded, 3 by default. Those
iterations pay for context creation, JIT, and cuBLAS or cuDNN autotuning, and
counting them distorts both the mean and the minimum.

A host span that measures exactly zero nanoseconds is discarded rather than
recorded. Two clock reads with real work between them cannot return the same
value, so a zero means the monotonic clock stalled, which happens on virtualized
hosts after a blocking call. One such sample would pin a label's minimum at zero
and stretch its jitter and histogram across a range nothing ever occupied. The
report says how many were dropped; GPU figures come from CUDA events and are
unaffected.

cadence does not lock clocks. Boost behaviour drifts from run to run, so pin
with `nvidia-smi -lgc <mhz>` before comparing two runs. The report records the
clock ceilings it saw either way.

## Overhead

Best difference over 7 x 50,000 scopes on an RTX A4000:

| | ns/scope | |
|---|---:|---|
| `CADENCE_KERNEL` | 3900 | two `cudaEventRecord` calls |
| `CADENCE_STAGE` | 2490 | one event, chained |
| `CADENCE_KERNEL`, 1 in 10 | 480 | sampled |
| `CADENCE_SCOPE` | 321 | two `steady_clock` reads |
| NVTX passthrough | ~20 | with no tool attached |

Nearly all of `CADENCE_KERNEL` is what the CUDA API charges for a timestamped
event rather than anything cadence does. The same call with
`cudaEventDisableTiming` costs 153 ns.
[docs/overhead.md](docs/overhead.md) has the measurements behind that, the two
ways to spend less, and what each one costs you in what you can still see.

In practice: wrap stages, not individual small kernels. Below roughly 50 µs of
GPU work per scope the instrumentation becomes a visible fraction of the result.

## Build and test

```sh
cmake -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The host tests, the disabled-build test, and the self-contained-header test need
no GPU and no CUDA toolkit. Device tests build when `nvcc` is found, and skip
themselves when no device answers. `examples/loop_saxpy.cu` produced the output
at the top of this page, and `benchmarks/overhead.cu` produced the table above.

## What it does not do

cadence measures elapsed time on a single GPU through the CUDA runtime API.
Hardware counters are `ncu`'s job and system-wide timelines are `nsys`'s. There
is no multi-GPU aggregation. Stream capture and CUDA graphs are not supported
yet, because a captured `cudaEventRecord` does not fit the deferred-flush model.

## License

MIT
