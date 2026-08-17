# Case study: instrumenting llama.cpp's CUDA backend

This study applies cadence to llama.cpp's CUDA backend while it runs a real
model. Unlike the repository's controlled benchmarks, llama.cpp uses CUDA graph
capture, kernel fusion, and a graph executor rather than a loop of named stages.

The results independently validate cadence's measured overhead, quantify the
cost of instrumenting at the wrong granularity, and expose three reporting
defects that were subsequently fixed and covered by regression tests.

## Setup

| | |
|---|---|
| llama.cpp | `4df29be`, built with `-DGGML_CUDA=ON`, Release, sm_86 |
| model | Qwen2.5-1.5B-Instruct, Q4_K_M, 1.04 GiB, 1.78 B params |
| GPU | RTX A4000 (sm_86), CUDA 12.9, clocks **not** pinned |
| harness | `llama-bench -n 128 -r 3`, so every figure is that tool's own throughput number |

The patch adds three things to `ggml/src/ggml-cuda/ggml-cuda.cu` and nothing
else: a host scope around `ggml_backend_cuda_graph_compute`, a device scope
around `cudaGraphLaunch`, and — in one mode — a device scope per graph node
labeled by `ggml_op_name(node->op)`. The mode is chosen at runtime by
`CADENCE_LLAMA_MODE`, so the same binary produces every row below.

## Graph-level results

Instrumenting only at the graph level, with llama.cpp's CUDA graphs on:

```
  label          scope     n    mean     p50     p95     max  jitter  distribution
  ────────────────────────────────────────────────────────────────────────────────
  graph-compute  host    369  8.30µs  8.14µs  9.12µs  18.9µs  11.2µs  █▂▂▂▂▂     ▂
  cuda-graph     device  367  3.93ms  3.93ms  4.00ms  4.04ms   144µs  ▂▅█▃▂▂▂▂▂ ▂▂
  cuda-graph     host    367  7.17µs  7.06µs  7.83µs  11.2µs  4.57µs  █▇▄▂▂▂▂    ▂
```

**The GPU measurement independently reproduces llama.cpp's own throughput.**
`cuda-graph` device mean is 3.93 ms; llama-bench reported 248.3 tok/s on the same
run, which is 4.03 ms per token. cadence is measuring the same thing llama.cpp is
measuring, from the other side, and the two agree to within the ~100 µs the host
spends outside the graph.

**Decode is 99.8% GPU time.** The CPU spends 8.30 µs per token issuing a
3.93 ms graph. The difference between the `device` and `host` rows identifies
the loop as compute-bound.

## CUDA graph capture behavior

llama.cpp captures its decode graph with `cudaStreamBeginCapture` and replays it
with `cudaGraphLaunch`. cadence documents that it refuses to record into a
capturing stream, because a `cudaEventRecord` issued during capture is baked into
the graph rather than executed — the event becomes permanently unreadable, and a
flush landing before the capture closes invalidates the capture outright.

With per-node instrumentation on, the report opens with:

```
  WARNING   535 scopes skipped during CUDA graph capture;
            instrument the graph launch instead
```

The guard correctly skips scopes recorded during graph capture. Instrumenting
the graph launch instead produces the `cuda-graph` row above.

There is a second, quieter consequence, and the `n` column is the only place it
shows up. Over 367 decodes, the per-node rows report `n` between 88 and 176.
After the graph is captured, `ggml_cuda_compute_forward` is **never called
again**: the nodes live inside the graph. So per-node instrumentation of
llama.cpp measures the handful of pre-capture evaluations and then goes quiet,
while the counts stay honest enough to say so. A tool that reported a mean
without a count would have looked entirely healthy.

## Validating instrumentation coverage

With CUDA graphs disabled (`GGML_CUDA_DISABLE_GRAPHS=1`), per-node scopes run on
every decode. The initial instrumentation wrapped
`ggml_cuda_compute_forward`, but the report accounted for only 381 µs of device
work within a 2.16 ms `graph-compute` host span.

The cause is in the loop above it:

```cpp
int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
if (nodes_to_skip != 0) {
    i += nodes_to_skip;
    continue;            // launches happened inside try_fuse
}
```

A fused group launches inside `ggml_cuda_try_fuse` and then skips
`ggml_cuda_compute_forward`. Moving the scope to cover both calls produced:

| op | scopes, obvious placement | scopes, fusion covered |
|---|---:|---:|
| `MUL_MAT` | 1,526 | 65,829 |
| `RMS_NORM` | 0 | 22,153 |
| `ROPE` | 10,872 | 21,764 |
| everything else | 23,511 | 23,511 |
| **total** | **35,909** | **133,257** |

The initial placement missed 97.7% of matrix multiplies and every `RMS_NORM`.
Instrumentation around a graph executor must cover fused fast paths, and its
component totals should be checked against the enclosing operation.

## Overhead cross-check

[docs/overhead.md](overhead.md) publishes 3390 ns per `CADENCE_KERNEL` scope from
a synthetic benchmark. llama.cpp offers a way to check that against real work: run
with graphs disabled, count the scopes, and read the throughput cost off
llama-bench.

| configuration | tok/s | ms/token | scopes/token | ns/scope |
|---|---:|---:|---:|---:|
| uninstrumented | 228.20 | 4.382 | 0 | — |
| per-op, obvious placement | 212.94 | 4.696 | 97 | **3227** |
| per-op, fusion covered | 182.43 | 5.482 | 361 | **3044** |

The two measurements use scope counts that differ by 3.7x, yet remain within
10% of the published figure and within 6% of each other. This is consistent with
the synthetic benchmark despite unpinned clocks and a different workload.

The results also quantify instrumentation granularity. At 361 scopes per token,
per-operation instrumentation reduces throughput by 20.1%. Graph-level
instrumentation uses three scopes per token and shows no measurable cost:

| configuration | tok/s | vs its own baseline |
|---|---:|---:|
| baseline, graphs on | 247.41 | — |
| instrumentation linked, mode off | 248.64 | +0.5% |
| graph-level scopes | 248.27 | +0.3% |
| baseline, graphs off | 228.20 | — |
| graph-level scopes, graphs off | 228.05 | −0.1% |
| per-op scopes, graphs off | 182.43 | **−20.1%** |

For context, CUDA graphs improve llama.cpp throughput by 8.4% on this model
(247.41 versus 228.20 tok/s). This is a workload result, not a cadence result.

## Defects identified and fixed

The graph workload exposed three cases not covered by the original small-stage
tests. Each defect is fixed and has a regression test verified against the
pre-fix behavior.

**1. The summary did not account for repeated labels.**
On the per-op run the report ends:

```
  device    205µs across 8 labels
  graph-compute  2.82ms, of which 7.3% is GPU work; 2.61ms is launch and synchronization
```

The workload is 99.8% GPU-bound. `WriteSummary` adds one mean per label, which is
correct only when each label occurs once per iteration; here `MUL_MAT` occurs 178
times per token. Weighting each mean by its occurrence rate gives 5.4 ms of
device work per iteration instead of 205 µs, reversing the classification.

*Fixed:* each mean is now weighted by `count / iterations`. The denominator is the
observation count of the single pure-host label, which is the loop body and ran
once per pass by construction. Without one there is no denominator, so the sum of
the means is now labeled as exactly that and the conclusion is withheld.

**2. The worst-iteration breakdown was unbounded.** With approximately 250 spans
per iteration, `WriteWorstIterations` produced lines several thousand characters
long.

*Fixed:* spans are folded by label — for example, `MUL_MAT 3.10ms x178` — sorted
widest first, and limited to eight entries with a `+N more` suffix.
Folding before cutting is what makes the cut useful; cutting 250 raw spans to
eight prints eight matrix multiplies.

**3. Iterations are ranked by the wrong span when the host scope does not enclose
the GPU work.** `IterationSpanMs` uses the longest host span as the iteration's
duration, documented on the assumption that a `CADENCE_SCOPE` wraps the loop body.
Here the host scope returns in 8 µs having queued 3.93 ms of GPU work, so the
"slowest iterations" are chosen by host jitter: the run picked an iteration with
an 18.9 µs host span and a perfectly ordinary 3.94 ms GPU span, while the actual
worst GPU iteration (4.04 ms) never appeared.

*Fixed:* the ranking now takes whichever of the longest host span and the total
device time is larger. Where the loop scope does enclose its GPU work it is the
larger of the two by construction, so the ordinary case is unchanged.

## Dynamic labels

`CADENCE_KERNEL` resolves its label through a function-local `static`, so the
handle is interned once per *call site*, not once per execution. Passing a label
that varies — `ggml_op_name(node->op)` — files every scope under whichever op ran
first, silently. The headers do say this (`ScopedHost`: "Interns on every
construction"; the macro: "the only form that resolves a label once per call site
rather than once per execution"), and the correct form for a runtime label is the
class directly:

```cpp
cadence::ScopedKernel scope(ggml_op_name(node->op), cuda_ctx->stream());
```

Use the class directly whenever a label changes between executions.

## Reproducing

The instrumentation patch is kept outside this repository because llama.cpp is
not a cadence dependency. It consists of the four insertions described above.

```sh
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
# add cadence's include dir to ggml-cuda, insert the four scopes, rebuild
CADENCE_LLAMA_MODE=graph ./build/bin/llama-bench -m model.gguf -n 128 -r 3
```

Clocks were not pinned (`nvidia-smi -lgc` needs root here), so read the ratios
rather than the absolutes. Every configuration was run against its own baseline
in the same session.
