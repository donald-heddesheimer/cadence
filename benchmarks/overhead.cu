// cadence benchmark: what does the instrumentation itself cost?

#include <cadence/cadence.h>

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace {
    constexpr int NUM_CHUNKS = 500;
    constexpr int NUM_SCOPES_PER_CHUNK = 100;  // One "loop iteration" of work.
    constexpr int NUM_SCOPES = NUM_CHUNKS * NUM_SCOPES_PER_CHUNK;
    constexpr int NUM_REPEATS = 7;

    // Scopes for the kernel-duration sweep, which runs far fewer of them because each one waits on real GPU work.
    constexpr int NUM_SPIN_CHUNKS = 20;
    constexpr int NUM_SPIN_SCOPES = NUM_SPIN_CHUNKS * NUM_SCOPES_PER_CHUNK;

    __global__ void Trivial(float* sink) {
        if (threadIdx.x == 0 && blockIdx.x == 0) *sink = 1.0f;
    }

    // A kernel with a duration, for showing how the two approaches respond to one. Busy-waiting on clock64() rather than doing arithmetic keeps the duration roughly independent of what the compiler decides to do with the body.
    __global__ void Spin(float* sink, long long cycles) {
        const long long start = clock64();
        while (clock64() - start < cycles) {
        }
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
    double BestNsPerScope(Baseline baseline, Instrumented instrumented, int scopes = NUM_SCOPES) {
        double best = 1e30;
        for (int repeat = 0; repeat < NUM_REPEATS; ++repeat) {
            const double bare = TimeNs(baseline);
            const double scoped = TimeNs(instrumented);
            best = std::min(best, (scoped - bare) / scopes);
        }
        return best;
    }

    // Absolute wall clock rather than a difference. The per-scope figures above answer "what does one scope cost"; this answers "what happened to the loop", which is a different question once an approach stops the CPU from running ahead of the GPU.
    template <typename Body>
    double BestTotalNs(Body body) {
        double best = 1e30;
        for (int repeat = 0; repeat < NUM_REPEATS; ++repeat) best = std::min(best, TimeNs(body));
        return best;
    }

    // Blocking comparison with events allocated once outside the loop.
    struct NaiveTimer {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        double totalMs = 0.0;  // Consumed so the elapsed-time query cannot be optimized away.

        NaiveTimer() {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
        }
        ~NaiveTimer() {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        void Begin(cudaStream_t stream) { cudaEventRecord(start, stream); }

        void End(cudaStream_t stream) {
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            float elapsedMs = 0.0f;
            cudaEventElapsedTime(&elapsedMs, start, stop);
            totalMs += elapsedMs;
        }
    };

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
    config.reportStream = nullptr;  // This benchmark prints its own numbers.
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

    // The hand-rolled comparison. Same launches, same events, but the elapsed time is read where it is taken rather than deferred, which means blocking on every scope.
    NaiveTimer naive;
    const double naiveNs = BestNsPerScope(launchOnly, [&] {
        for (int chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
            for (int i = 0; i < NUM_SCOPES_PER_CHUNK; ++i) {
                naive.Begin(stream);
                Trivial<<<1, 32, 0, stream>>>(sink);
                naive.End(stream);
            }
            cudaStreamSynchronize(stream);
        }
    });

    // What each approach does to the loop as a whole, rather than to one scope. A per-scope cost understates blocking: the cost of a synchronize is not what it spends, it is the overlap it gives up.
    const double bareTotalNs = BestTotalNs(launchOnly);
    const double cadenceTotalNs = BestTotalNs([&] {
        for (int chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
            for (int i = 0; i < NUM_SCOPES_PER_CHUNK; ++i) {
                CADENCE_KERNEL("trivial-total", stream);
                Trivial<<<1, 32, 0, stream>>>(sink);
            }
            cudaStreamSynchronize(stream);
            cadence::Flush();
        }
        cadence::Reset();
    });
    const double naiveTotalNs = BestTotalNs([&] {
        for (int chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
            for (int i = 0; i < NUM_SCOPES_PER_CHUNK; ++i) {
                naive.Begin(stream);
                Trivial<<<1, 32, 0, stream>>>(sink);
                naive.End(stream);
            }
            cudaStreamSynchronize(stream);
        }
    });

    // The same comparison over a kernel with a real duration, because the trivial kernel above flatters the blocking approach: it is cheap to wait for something that takes no time.
    //
    // Compare deferred and blocking measurement over nontrivial GPU work.
    constexpr long long SPIN_CYCLES = 31200;  // ~20 us at the A4000's 1.56 GHz boost clock.
    const auto spinOnly = [&] {
        for (int chunk = 0; chunk < NUM_SPIN_CHUNKS; ++chunk) {
            for (int i = 0; i < NUM_SCOPES_PER_CHUNK; ++i) Spin<<<1, 32, 0, stream>>>(sink, SPIN_CYCLES);
            cudaStreamSynchronize(stream);
        }
    };
    spinOnly();
    const double spinKernelNs = BestTotalNs(spinOnly) / NUM_SPIN_SCOPES;

    const double spinCadenceNs = BestNsPerScope(spinOnly, [&] {
        for (int chunk = 0; chunk < NUM_SPIN_CHUNKS; ++chunk) {
            for (int i = 0; i < NUM_SCOPES_PER_CHUNK; ++i) {
                CADENCE_KERNEL("spin", stream);
                Spin<<<1, 32, 0, stream>>>(sink, SPIN_CYCLES);
            }
            cudaStreamSynchronize(stream);
            cadence::Flush();
        }
        cadence::Reset();
    }, NUM_SPIN_SCOPES);

    const double spinNaiveNs = BestNsPerScope(spinOnly, [&] {
        for (int chunk = 0; chunk < NUM_SPIN_CHUNKS; ++chunk) {
            for (int i = 0; i < NUM_SCOPES_PER_CHUNK; ++i) {
                naive.Begin(stream);
                Spin<<<1, 32, 0, stream>>>(sink, SPIN_CYCLES);
                naive.End(stream);
            }
            cudaStreamSynchronize(stream);
        }
    }, NUM_SPIN_SCOPES);

    std::printf("scopes per measurement: %d (best difference of %d)\n\n", NUM_SCOPES, NUM_REPEATS);
    std::printf("CADENCE_KERNEL          %8.1f ns/scope\n", kernelNs);
    std::printf("CADENCE_STAGE           %8.1f ns/scope   (%.0f%% of CADENCE_KERNEL)\n", stageNs,
                100.0 * stageNs / kernelNs);
    std::printf("CADENCE_KERNEL, 1 in 10 %8.1f ns/scope   (%.0f%% of CADENCE_KERNEL)\n", sampledNs,
                100.0 * sampledNs / kernelNs);
    std::printf("CADENCE_SCOPE           %8.1f ns/scope\n", hostNs);
    std::printf("  NVTX passthrough      %8.1f ns/scope\n", nvtxNs - kernelNs);
    std::printf("hand-rolled cudaEvent   %8.1f ns/scope   (%.1fx CADENCE_KERNEL)\n", naiveNs, naiveNs / kernelNs);

    std::printf("\nwhole-loop wall clock, %d launches:\n", NUM_SCOPES);
    std::printf("  uninstrumented        %8.2f ms\n", bareTotalNs * 1e-6);
    std::printf("  CADENCE_KERNEL        %8.2f ms   (%.2fx)\n", cadenceTotalNs * 1e-6, cadenceTotalNs / bareTotalNs);
    std::printf("  hand-rolled           %8.2f ms   (%.2fx)\n", naiveTotalNs * 1e-6, naiveTotalNs / bareTotalNs);

    std::printf("\nover a %.1f us kernel, %d scopes:\n", spinKernelNs * 1e-3, NUM_SPIN_SCOPES);
    std::printf("  CADENCE_KERNEL        %8.1f ns/scope\n", spinCadenceNs);
    std::printf("  hand-rolled           %8.1f ns/scope   (%.1fx CADENCE_KERNEL)\n", spinNaiveNs,
                spinNaiveNs / spinCadenceNs);

    cudaStreamDestroy(stream);
    cudaFree(sink);
    return 0;
}
