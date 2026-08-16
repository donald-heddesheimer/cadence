# Overhead

Every number below is the best difference over 7 x 50,000 scopes on an RTX A4000
(sm_86), produced by [`benchmarks/overhead.cu`](../benchmarks/overhead.cu).

| | ns/scope | share of `CADENCE_KERNEL` |
|---|---:|---:|
| `CADENCE_KERNEL` | 3390 | 100% |
| `CADENCE_STAGE` | 2410 | 71% |
| `CADENCE_KERNEL`, `sampleEvery = 10` | 367 | 11% |
| `CADENCE_SCOPE` | 330 | |
| NVTX passthrough, no tool attached | ~10 | |

## The correctness guards

About 200 ns of every device scope buys two things that cannot be skipped:

- `cudaStreamIsCapturing`, at 39.5 ns per call and called on both ends of a
  scope, which keeps cadence from recording into a stream capturing a CUDA
  graph. Recording anyway makes the events permanently unreadable, and a flush
  landing before the capture closes invalidates the capture itself and hands the
  application a null graph.
- `cudaGetDevice`, at 20.6 ns, which keeps events paired with the GPU that
  created them. Events are device-local and neither `cudaEventRecord` against
  another device's stream nor `cudaEventElapsedTime` across a device boundary is
  legal.

Measured against the same benchmark before the guards existed, the cost is +160
to +230 ns per device scope depending on the run, or roughly 5 to 7%. Both
failures they prevent are silent, which is why neither is configurable.
`CADENCE_SCOPE` touches no CUDA and pays neither.

## Where the time goes

Almost all of `CADENCE_KERNEL` is two `cudaEventRecord` calls at roughly 1.4 µs
each. That cost belongs to the CUDA API rather than to cadence, and three
separate measurements say so:

- It does not change with burst size. One, two, four, and eight consecutive records cost 1459, 1382, 1399, and 1406 ns. Batching buys nothing.
- It does not change with launch-queue depth. Zero, one, four, and sixteen pending launches make no difference.
- It drops to 153 ns with `cudaEventDisableTiming`. The remaining 1250 ns or so is the timestamp itself, not the ioctl.

Two of those records account for about 72% of a device scope. The rest of what
cadence does, meaning the event pool, the record append, and the flush, adds up
to a few percent, and has been optimized far enough that further work there is
not worth the complexity. For scale, on the same machine an uncontended mutex is
18 ns, `cudaEventSynchronize` on an already-complete event is 142 ns, and
`cudaEventElapsedTime` is 249 ns.

So the only real lever is recording fewer events. That is what the two opt-in
modes below do, and neither is free.

## `CADENCE_STAGE`: one event instead of two

A stage reuses the previous stage's stop event as its own start, so a pipeline of
K stages records K+1 events rather than 2K.

```cpp
{ CADENCE_STAGE("detect",  stream); Detect<<<...>>>(...);  }
{ CADENCE_STAGE("track",   stream); Track<<<...>>>(...);   }
{ CADENCE_STAGE("fuse",    stream); Fuse<<<...>>>(...);    }
```

What it costs you: a stage is timed from the end of the previous stage on that
stream, not from its own first launch. Anything you enqueue between two stages
gets charged to the later one.

Use it when every launch on the stream sits inside some stage, which is the
normal case for a pipeline you built deliberately. Use `CADENCE_KERNEL` when
that is not true, or when a number has to stand on its own without reference to
its neighbours.

Two device tests pin the trade-off in both directions,
`TestChainingChargesGapsToTheNextStage` and `TestPairedScopeIgnoresGaps`, so the
behaviour cannot drift silently.

On a realistic 5-stage, 20 µs-per-stage loop, chaining measured 34% less total
instrumentation cost than the paired path.

## `sampleEvery`: one iteration in N

```cpp
cadence::Config cfg;
cfg.sampleEvery = 10;   // or CADENCE_SAMPLE=10 in the environment
cadence::Configure(cfg);
```

This divides the cost rather than shaving it, and it pays for that in coverage.
You get the distribution of the iterations it happened to look at, so a
once-a-minute outlier is something you may simply miss. If you are hunting a
rare spike, this is the wrong knob.

The report header records the rate so the counts stay interpretable. Without it, a
10,000-iteration run reporting 1,000 samples reads as a bug rather than a
setting.

## Host-side timing

`CADENCE_SCOPE` is two `steady_clock` reads plus a record append. The 330 ns
above is inflated by the test machine, a Xen VM where `steady_clock::now()`
costs about 151 ns against 20 to 30 ns on a `tsc` clocksource. On bare metal
expect well under 100 ns. Measure on your own hardware before budgeting.

## Approaches measured and rejected

Kept here so they do not get proposed again without new evidence.

**`rdtsc` instead of `steady_clock`.** The idea works against itself. TSC only
pays off where `steady_clock` is slow, meaning virtualized hosts, which is
exactly where TSC is least trustworthy. On bare metal `steady_clock` is already
a 20 ns vDSO read, so the best case is a saving of about 10 ns per read.

**A timestamp kernel, or `cudaLaunchHostFunc`.** An empty kernel launch costs
2030 ns, worse than the 1400 ns record it would replace.

**Dropping the global mutex for a multithreaded speedup.** cadence now uses
thread-local record buffers with per-thread event caches, which is a genuine
architectural improvement, but it is not presented as a benchmark result because
it could not be measured. The CUDA driver's own launch serialization dominates,
and baselines swing between 4300 and 7300 ns at four threads.

**CUPTI** is the one remaining path that takes events off the hot path entirely.
The driver already timestamps every kernel, so an activity-API backend would need
no instrumentation events at all. That is the long-term direction rather than a
micro-optimization.

## Reproducing these numbers

The methodology matters more than it looks. Measuring the baseline and the
instrumented loop in separate phases lets GPU clocks drift between them, which
produced 2x swings from run to run. The benchmark instead interleaves both
inside each repeat and takes the minimum of the differences, which reproduces to
within about 1%.

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
nvidia-smi -lgc 1560          # pin clocks first
./build/benchmarks/cadence_overhead
```
