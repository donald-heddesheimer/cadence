// cadence benchmark — what does the instrumentation itself cost?
//
// A profiler you cannot trust to stay out of the way is worse than no profiler.
// This measures the host-side cost cadence adds per scope, so the numbers in
// the README are measured rather than hoped for.
//
// Method. The kernel is trivial and the loop is long, so the CPU is the
// bottleneck and what is being timed is the instrumentation rather than the
// GPU. Baseline and instrumented run back to back inside each repeat and the
// minimum of their differences is kept: measuring them in separate phases lets
// the GPU clock drift between the two, which on a boosting part is worth more
// than the thing being measured. The minimum is the right statistic because
// scheduling noise only ever adds.
//
// The loop is chunked and the baseline synchronizes on the same chunk boundary
// as the instrumented version, because that is how the library is meant to be
// used: flush once per loop iteration, not once per run. It also matters for
// the number -- a run that never flushes never recycles an event, so it would
// measure cudaEventCreate instead of cadence. The instrumented figure therefore
// includes its share of the deferred flush.

#include <cadence/cadence.h>

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace {

constexpr int NUM_CHUNKS = 500;
constexpr int NUM_SCOPES_PER_CHUNK = 100;  // One "loop iteration" of work.
constexpr int NUM_SCOPES = NUM_CHUNKS * NUM_SCOPES_PER_CHUNK;
constexpr int NUM_REPEATS = 7;

__global__ void Trivial(float* sink) {
  if (threadIdx.x == 0 && blockIdx.x == 0) *sink = 1.0f;
}

template <typename Body>
double TimeNs(Body body) {
  const auto start = std::chrono::steady_clock::now();
  body();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::nano>(end - start).count();
}

template <typename Baseline, typename Instrumented>
double BestNsPerScope(Baseline baseline, Instrumented instrumented) {
  double best = 1e30;
  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat) {
    const double bare = TimeNs(baseline);
    const double scoped = TimeNs(instrumented);
    best = std::min(best, (scoped - bare) / NUM_SCOPES);
  }
  return best;
}

}  // namespace

int main() {
  float* sink = nullptr;
  if (cudaMalloc(&sink, sizeof(float)) != cudaSuccess) {
    std::fprintf(stderr, "no CUDA device available\n");
    return 1;
  }
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  cadence::Config config;
  config.outputPath.clear();  // This benchmark reports to stdout, not to a file.
  config.warmupIterations = 0;
  config.nvtxEnabled = false;
  config.writeOnExit = false;
  cadence::Configure(config);

  // Baseline: the launches and the same synchronization cadence, no cadence.
  const auto launchOnly = [&] {
    for (int chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
      for (int i = 0; i < NUM_SCOPES_PER_CHUNK; ++i) Trivial<<<1, 32, 0, stream>>>(sink);
      cudaStreamSynchronize(stream);
    }
  };
  launchOnly();  // Get the clocks up before anything is recorded.

  const double kernelNs = BestNsPerScope(launchOnly, [&] {
    for (int chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
      for (int i = 0; i < NUM_SCOPES_PER_CHUNK; ++i) {
        CADENCE_KERNEL("trivial", stream);
        Trivial<<<1, 32, 0, stream>>>(sink);
      }
      cudaStreamSynchronize(stream);
      cadence::Flush();
    }
    cadence::Reset();
  });

  const double stageNs = BestNsPerScope(launchOnly, [&] {
    for (int chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
      for (int i = 0; i < NUM_SCOPES_PER_CHUNK; ++i) {
        CADENCE_STAGE("trivial-stage", stream);
        Trivial<<<1, 32, 0, stream>>>(sink);
      }
      cudaStreamSynchronize(stream);
      cadence::Flush();
    }
    cadence::Reset();
  });

  // One scope in ten measured. The other nine cost an atomic increment and a branch, which is the only knob that divides the cudaEventRecord cost rather than shaving at it.
  config.sampleEvery = 10;
  cadence::Configure(config);
  const double sampledNs = BestNsPerScope(launchOnly, [&] {
    for (int chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
      for (int i = 0; i < NUM_SCOPES_PER_CHUNK; ++i) {
        CADENCE_KERNEL("trivial-sampled", stream);
        Trivial<<<1, 32, 0, stream>>>(sink);
      }
      cudaStreamSynchronize(stream);
      cadence::Flush();
    }
    cadence::Reset();
  });
  config.sampleEvery = 1;
  cadence::Configure(config);

  // Instrumented with NVTX on, and no tool attached. The interesting number is the difference against the run above.
  config.nvtxEnabled = true;
  cadence::Configure(config);
  const double nvtxNs = BestNsPerScope(launchOnly, [&] {
    for (int chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
      for (int i = 0; i < NUM_SCOPES_PER_CHUNK; ++i) {
        CADENCE_KERNEL("trivial-nvtx", stream);
        Trivial<<<1, 32, 0, stream>>>(sink);
      }
      cudaStreamSynchronize(stream);
      cadence::Flush();
    }
    cadence::Reset();
  });
  config.nvtxEnabled = false;
  cadence::Configure(config);

  // The host scope on its own, with no launch underneath it.
  const auto emptyLoop = [&] {
    volatile int counter = 0;
    for (int i = 0; i < NUM_SCOPES; ++i) counter = counter + 1;
  };
  const double hostNs = BestNsPerScope(emptyLoop, [&] {
    volatile int counter = 0;
    for (int chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
      for (int i = 0; i < NUM_SCOPES_PER_CHUNK; ++i) {
        CADENCE_SCOPE("host-scope");
        counter = counter + 1;
      }
      cadence::Flush();
    }
    cadence::Reset();
  });

  std::printf("scopes per measurement: %d (best difference of %d)\n\n", NUM_SCOPES, NUM_REPEATS);
  std::printf("CADENCE_KERNEL          %8.1f ns/scope\n", kernelNs);
  std::printf("CADENCE_STAGE           %8.1f ns/scope   (%.0f%% of CADENCE_KERNEL)\n", stageNs,
              100.0 * stageNs / kernelNs);
  std::printf("CADENCE_KERNEL, 1 in 10 %8.1f ns/scope   (%.0f%% of CADENCE_KERNEL)\n", sampledNs,
              100.0 * sampledNs / kernelNs);
  std::printf("CADENCE_SCOPE           %8.1f ns/scope\n", hostNs);
  std::printf("  NVTX passthrough      %8.1f ns/scope\n", nvtxNs - kernelNs);

  cudaStreamDestroy(stream);
  cudaFree(sink);
  return 0;
}
