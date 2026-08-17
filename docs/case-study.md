# Case study: instrumenting llama.cpp's CUDA backend

Everything else in this repository is measured against a benchmark written to be
measured. This is cadence pointed at code it did not grow up with: llama.cpp's
CUDA backend, unmodified except for the instrumentation, running a real model.

The point was to find out what happens when the library meets a codebase with its
own ideas — CUDA graph capture, kernel fusion, a graph executor rather than a
loop of named stages. Three things came out of it. One validates a number this
repository has been publishing. One is a fact about llama.cpp. One is a defect in
cadence that produces a confidently wrong conclusion, and it is written up here
in full because a case study that only finds good news is not worth reading.

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
labelled by `ggml_op_name(node->op)`. The mode is chosen at runtime by
`CADENCE_LLAMA_MODE`, so the same binary produces every row below.

## What the report says

Instrumenting only at the graph level, with llama.cpp's CUDA graphs on:

```
  label          scope     n    mean     p50     p95     max  jitter  distribution
  ────────────────────────────────────────────────────────────────────────────────
  graph-compute  host    369  8.30µs  8.14µs  9.12µs  18.9µs  11.2µs  █▂▂▂▂▂     ▂
  cuda-graph     device  367  3.93ms  3.93ms  4.00ms  4.04ms   144µs  ▂▅█▃▂▂▂▂▂ ▂▂
  cuda-graph     host    367  7.17µs  7.06µs  7.83µs  11.2µs  4.57µs  █▇▄▂▂▂▂    ▂
```

Two things are worth reading off it.

**The GPU measurement independently reproduces llama.cpp's own throughput.**
`cuda-graph` device mean is 3.93 ms; llama-bench reported 248.3 tok/s on the same
run, which is 4.03 ms per token. cadence is measuring the same thing llama.cpp is
measuring, from the other side, and the two agree to within the ~100 µs the host
spends outside the graph.

**Decode is 99.8% GPU.** The CPU spends 8.30 µs per token issuing a 3.93 ms graph.
That is the `device` and `host` row pair doing exactly what the README says they
are for: when they diverge this far, the loop is compute-bound and no amount of
CPU-side work will help it.

## The capture guard fires, on somebody else's code

llama.cpp captures its decode graph with `cudaStreamBeginCapture` and replays it
with `cudaGraphLaunch`. cadence documents that it refuses to record into a
capturing stream, because a `cudaEventRecord` issued during capture is baked into
the graph rather than executed — the event becomes permanently unreadable, and a
flush landing before the capture closes invalidates the capture outright.

With per-node instrumentation on, the report opens with:

```
  WARNING   535 scope(s) skipped -- their stream was capturing into a CUDA graph,
            which cannot carry timing events; wrap the graph launch instead
```

This is the guard working as designed against real third-party code that really
does use CUDA graphs, and the advice in that message — *wrap the graph launch
instead* — is exactly what the `cuda-graph` row above is.

There is a second, quieter consequence, and the `n` column is the only place it
shows up. Over 367 decodes, the per-node rows report `n` between 88 and 176.
After the graph is captured, `ggml_cuda_compute_forward` is **never called
again**: the nodes live inside the graph. So per-node instrumentation of
llama.cpp measures the handful of pre-capture evaluations and then goes quiet,
while the counts stay honest enough to say so. A tool that reported a mean
without a count would have looked entirely healthy.

## The measurement that caught my own mistake

With CUDA graphs disabled (`GGML_CUDA_DISABLE_GRAPHS=1`), per-node scopes run on
every decode and the per-op distribution becomes real. The first version of that
patch put the scope around `ggml_cuda_compute_forward`, which is the obvious line.
The report then claimed 381 µs of device work per iteration against a
`graph-compute` host span of 2.16 ms — most of the GPU time was unaccounted for.

The cause is in the loop above it:

```cpp
int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
if (nodes_to_skip != 0) {
    i += nodes_to_skip;
    continue;            // launches happened inside try_fuse
}
```

A fused group launches its work inside `ggml_cuda_try_fuse` and then `continue`s
straight past `ggml_cuda_compute_forward`. A scope on the obvious line sees none
of it. Moving the scope to cover both calls:

| op | scopes, obvious placement | scopes, fusion covered |
|---|---:|---:|
| `MUL_MAT` | 1,526 | 65,829 |
| `RMS_NORM` | 0 | 22,153 |
| `ROPE` | 10,872 | 21,764 |
| everything else | 23,511 | 23,511 |
| **total** | **35,909** | **133,257** |

The first placement missed 97.7% of the matrix multiplies and every single
`RMS_NORM` — an entire op type that simply was not in the report. The lesson is
not about cadence; it is that instrumenting somebody else's graph executor at the
line that looks right will quietly miss its fast path, and the only defence is
checking whether the parts add up to the whole.

## The overhead number holds up outside its own benchmark

[docs/overhead.md](overhead.md) publishes 3390 ns per `CADENCE_KERNEL` scope from
a synthetic benchmark. llama.cpp offers a way to check that against real work: run
with graphs disabled, count the scopes, and read the throughput cost off
llama-bench.

| configuration | tok/s | ms/token | scopes/token | ns/scope |
|---|---:|---:|---:|---:|
| uninstrumented | 228.20 | 4.382 | 0 | — |
| per-op, obvious placement | 212.94 | 4.696 | 97 | **3227** |
| per-op, fusion covered | 182.43 | 5.482 | 361 | **3044** |

Two measurements, taken at scope counts that differ by 3.7x, on somebody else's
kernels, land within 10% of the published figure and within 6% of each other.
That is about as good as an unpinned-clock cross-check gets, and it is the first
evidence in this repository that the number means anything outside the benchmark
that produced it.

It also settles the README's advice with a real example. "Wrap stages, not
individual small kernels" costs 20.1% of throughput here when taken literally at
361 scopes per token — while the graph-level instrumentation, three scopes per
token, is free:

| configuration | tok/s | vs its own baseline |
|---|---:|---:|
| baseline, graphs on | 247.41 | — |
| instrumentation linked, mode off | 248.64 | +0.5% |
| graph-level scopes | 248.27 | +0.3% |
| baseline, graphs off | 228.20 | — |
| graph-level scopes, graphs off | 228.05 | −0.1% |
| per-op scopes, graphs off | 182.43 | **−20.1%** |

(Incidentally: CUDA graphs are worth 8.4% to llama.cpp on this model, 247.41
against 228.20 tok/s. Not a cadence result, but it fell out of the same runs.)

## Three defects this turned up in cadence

None of these are visible on a loop with a handful of stages, which is the only
shape the tests and benchmarks had ever covered. All three are described below as
they were found, and all three are now fixed — each with a regression test that
was checked to fail with its own fix reverted.

**1. The summary line states a conclusion that is the opposite of the truth.**
On the per-op run the report ends:

```
  device    205µs across 8 label(s)
  graph-compute  2.82ms, of which 7.3% is GPU work; 2.61ms is launch and synchronization
```

The workload is 99.8% GPU-bound. `WriteSummary` adds one mean per label, which is
correct only when each label occurs once per iteration; here `MUL_MAT` occurs 178
times per token. Weighting each mean by its actual occurrence rate gives 5.4 ms of
device work per iteration, not 205 µs. The line is not merely imprecise — it
confidently reports the single most important fact about this workload backwards.

*Fixed:* each mean is now weighted by `count / iterations`. The denominator is the
observation count of the single pure-host label, which is the loop body and ran
once per pass by construction. Without one there is no denominator, so the sum of
the means is now labelled as exactly that and the conclusion is withheld.

**2. The worst-iteration breakdown is unbounded.** With ~250 spans in an
iteration, `WriteWorstIterations` prints all of them on one line: a single
unreadable paragraph several thousand characters long, three times over.

*Fixed:* spans are folded by label — `MUL_MAT 3.10ms x178`, which is the number
worth reading anyway — sorted widest first, and cut to eight with a `+N more`.
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

## One thing that is not a defect, but caught me anyway

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

Documented, and still the first thing I got wrong. Worth knowing before you reach
for a dynamic label.

## Reproducing

The patch is a script rather than a diff so it survives llama.cpp moving lines
around, and so it is obvious what was inserted where. It is not in this
repository — llama.cpp is not a dependency of cadence and should not become one —
but it is four insertions, all quoted above.

```sh
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
# add cadence's include dir to ggml-cuda, insert the four scopes, rebuild
CADENCE_LLAMA_MODE=graph ./build/bin/llama-bench -m model.gguf -n 128 -r 3
```

Clocks were not pinned (`nvidia-smi -lgc` needs root here), so read the ratios
rather than the absolutes. Every configuration was run against its own baseline
in the same session.
