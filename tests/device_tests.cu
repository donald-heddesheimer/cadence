// cadence device tests: the parts that need a real GPU behind them.
//
// Exits 77 when no device answers, which CTest is told to read as "skipped".

#include <cadence/cadence.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {
    int failures = 0;

    void Check(bool condition, const std::string& what) {
        std::printf("[%s] %s\n", condition ? " ok " : "FAIL", what.c_str());
        if (!condition) ++failures;
    }

    void CheckNear(double actual, double expected, double tolerance, const std::string& what) {
        const bool ok = actual >= expected * (1.0 - tolerance) && actual <= expected * (1.0 + tolerance);
        std::printf("[%s] %s (got %.4f, expected %.4f +/- %.0f%%)\n", ok ? " ok " : "FAIL", what.c_str(),
                    actual, expected, tolerance * 100.0);
        if (!ok) ++failures;
    }

    // Generate controllable GPU work; tests compare ratios rather than wall time.
    __global__ void Spin(float* sink, int iterations) {
        float value = 0.0f;
        for (int i = 0; i < iterations; ++i) value = fmaf(value, 1.0000001f, 1e-7f);
        if (threadIdx.x == 0 && blockIdx.x == 0) *sink = value;
    }

    constexpr int SPIN_UNIT = 20000;
    constexpr int NUM_BLOCKS = 256;
    constexpr int NUM_THREADS = 256;

    // Allow for midpoint error when placing GPU spans on the host clock.
    constexpr std::int64_t TRACE_TOLERANCE_NS = 50000;

    void BurnHostMicroseconds(int microseconds) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(microseconds);
        while (std::chrono::steady_clock::now() < until) {
        }
    }

    double DeviceMs(const std::vector<cadence::Stats>& snapshot, const std::string& label) {
        for (const cadence::Stats& row : snapshot) {
            if (row.label == label && row.kind == cadence::ScopeKind::Device) return row.meanMs;
        }
        return -1.0;
    }

    std::size_t CountOf(const std::vector<cadence::Stats>& snapshot, const std::string& label,
                        cadence::ScopeKind kind) {
        for (const cadence::Stats& row : snapshot) {
            if (row.label == label && row.kind == kind) return row.count;
        }
        return 0;
    }

    void ResetLibrary(unsigned warmup, unsigned sampleEvery) {
        cadence::Config config;
        config.reportStream = nullptr;
        config.warmupIterations = warmup;
        config.nvtxEnabled = false;
        config.writeOnExit = false;
        config.sampleEvery = sampleEvery;
        cadence::Configure(config);
        cadence::Reset();
    }

    // A paired scope should report the time of the work it brackets, so doubling the work should double the number.
    void TestPairedScopeTracksWork(float* sink, cudaStream_t stream) {
        ResetLibrary(2, 1);
        for (int i = 0; i < 30; ++i) {
            {
                CADENCE_KERNEL("paired-1x", stream);
                Spin<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(sink, SPIN_UNIT);
            }
            {
                CADENCE_KERNEL("paired-2x", stream);
                Spin<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(sink, SPIN_UNIT * 2);
            }
            cadence::Flush();
        }
        const auto snapshot = cadence::Snapshot();
        const double single = DeviceMs(snapshot, "paired-1x");
        const double doubled = DeviceMs(snapshot, "paired-2x");
        Check(single > 0.0, "paired scope produced a device measurement");
        CheckNear(doubled / single, 2.0, 0.20, "paired scope scales with work");
        Check(CountOf(snapshot, "paired-1x", cadence::ScopeKind::Device) == 28,
              "paired scope discarded exactly the warmup iterations");
    }

    // The same shape through the chained scope. Every launch on the stream is inside a stage, so chaining has no gap to absorb and should agree with the paired numbers.
    void TestChainedStagesTrackWork(float* sink, cudaStream_t stream) {
        ResetLibrary(2, 1);
        for (int i = 0; i < 30; ++i) {
            {
                CADENCE_STAGE("stage-1x", stream);
                Spin<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(sink, SPIN_UNIT);
            }
            {
                CADENCE_STAGE("stage-2x", stream);
                Spin<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(sink, SPIN_UNIT * 2);
            }
            cadence::Flush();
        }
        const auto snapshot = cadence::Snapshot();
        const double single = DeviceMs(snapshot, "stage-1x");
        const double doubled = DeviceMs(snapshot, "stage-2x");
        Check(single > 0.0, "chained stage produced a device measurement");
        CheckNear(doubled / single, 2.0, 0.20, "chained stage scales with work");
        Check(CountOf(snapshot, "stage-1x", cadence::ScopeKind::Device) == 28,
              "chained stage discarded exactly the warmup iterations");
    }

    // Chained stages charge uninstrumented gaps to the following stage.
    void TestChainingChargesGapsToTheNextStage(float* sink, cudaStream_t stream) {
        ResetLibrary(2, 1);
        for (int i = 0; i < 30; ++i) {
            {
                CADENCE_STAGE("gap-first", stream);
                Spin<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(sink, SPIN_UNIT);
            }
            // Deliberately outside any scope, which is exactly what the chained mode asks you not to do.
            Spin<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(sink, SPIN_UNIT);
            {
                CADENCE_STAGE("gap-second", stream);
                Spin<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(sink, SPIN_UNIT);
            }
            cadence::Flush();
        }
        const auto snapshot = cadence::Snapshot();
        const double first = DeviceMs(snapshot, "gap-first");
        const double second = DeviceMs(snapshot, "gap-second");
        CheckNear(second / first, 2.0, 0.20, "an uninstrumented launch lands on the following stage");
    }

    // A paired scope in the same situation ignores the gap entirely: it brackets its own work and nothing else.
    void TestPairedScopeIgnoresGaps(float* sink, cudaStream_t stream) {
        ResetLibrary(2, 1);
        for (int i = 0; i < 30; ++i) {
            {
                CADENCE_KERNEL("nogap-first", stream);
                Spin<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(sink, SPIN_UNIT);
            }
            Spin<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(sink, SPIN_UNIT);
            {
                CADENCE_KERNEL("nogap-second", stream);
                Spin<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(sink, SPIN_UNIT);
            }
            cadence::Flush();
        }
        const auto snapshot = cadence::Snapshot();
        const double first = DeviceMs(snapshot, "nogap-first");
        const double second = DeviceMs(snapshot, "nogap-second");
        CheckNear(second / first, 1.0, 0.20, "a paired scope is unaffected by an uninstrumented launch");
    }

    // Shared chain events must return to the pool exactly once.
    void TestEventPoolSettles(float* sink, cudaStream_t stream) {
        ResetLibrary(0, 1);
        const auto run = [&](int iterations) {
            for (int i = 0; i < iterations; ++i) {
                {
                    CADENCE_STAGE("pool-a", stream);
                    Spin<<<8, 64, 0, stream>>>(sink, 64);
                }
                {
                    CADENCE_STAGE("pool-b", stream);
                    Spin<<<8, 64, 0, stream>>>(sink, 64);
                }
                {
                    CADENCE_KERNEL("pool-c", stream);
                    Spin<<<8, 64, 0, stream>>>(sink, 64);
                }
                cadence::Flush();
            }
        };
        run(200);
        const std::size_t afterWarmup = cadence::detail::Registry::Instance().LiveEventCount();
        run(2000);
        const std::size_t afterLong = cadence::detail::Registry::Instance().LiveEventCount();
        Check(afterLong == afterWarmup, "live event count is unchanged after 10x the iterations (" +
                                            std::to_string(afterWarmup) + " events)");
        Check(cadence::FailedRecordCount() == 0, "no records dropped across 2200 iterations");
        const auto snapshot = cadence::Snapshot();
        Check(CountOf(snapshot, "pool-a", cadence::ScopeKind::Device) == 2200,
              "every chained iteration produced exactly one sample");
    }

    // Interleaved streams must keep independent stage chains. Compare each
    // chained result with a paired scope under the same contention.
    void TestChainsArePerStream(float* sink, cudaStream_t first, cudaStream_t second) {
        const auto runLoop = [&](bool chained) {
            for (int i = 0; i < 30; ++i) {
                if (chained) {
                    {
                        CADENCE_STAGE("chained-a", first);
                        Spin<<<32, 128, 0, first>>>(sink, SPIN_UNIT);
                    }
                    {
                        CADENCE_STAGE("chained-b", second);
                        Spin<<<32, 128, 0, second>>>(sink, SPIN_UNIT * 3);
                    }
                } else {
                    {
                        CADENCE_KERNEL("paired-a", first);
                        Spin<<<32, 128, 0, first>>>(sink, SPIN_UNIT);
                    }
                    {
                        CADENCE_KERNEL("paired-b", second);
                        Spin<<<32, 128, 0, second>>>(sink, SPIN_UNIT * 3);
                    }
                }
                cudaStreamSynchronize(first);
                cudaStreamSynchronize(second);
                cadence::Flush();
            }
        };

        ResetLibrary(2, 1);
        runLoop(false);
        runLoop(true);
        const auto snapshot = cadence::Snapshot();
        const double pairedFirst = DeviceMs(snapshot, "paired-a");
        const double pairedSecond = DeviceMs(snapshot, "paired-b");
        const double chainedFirst = DeviceMs(snapshot, "chained-a");
        const double chainedSecond = DeviceMs(snapshot, "chained-b");
        Check(pairedFirst > 0.0 && pairedSecond > 0.0 && chainedFirst > 0.0 && chainedSecond > 0.0,
              "both streams produced measurements both ways");
        CheckNear(chainedFirst, pairedFirst, 0.25, "chained stage on stream 1 matches its paired scope");
        CheckNear(chainedSecond, pairedSecond, 0.25,
                  "chained stage on stream 2 matches its paired scope");
    }

    void TestSamplingKeepsEveryNth(float* sink, cudaStream_t stream) {
        ResetLibrary(0, 4);
        for (int i = 0; i < 100; ++i) {
            {
                CADENCE_KERNEL("sampled", stream);
                Spin<<<8, 64, 0, stream>>>(sink, 64);
            }
            cadence::Flush();
        }
        const auto snapshot = cadence::Snapshot();
        Check(CountOf(snapshot, "sampled", cadence::ScopeKind::Device) == 25,
              "sampleEvery=4 measured 25 of 100 iterations");
        ResetLibrary(0, 1);
    }

    // Sampling keeps or drops an entire chained-stage sequence.
    void TestStageSamplingThinsWholeChains(float* sink, cudaStream_t stream) {
        ResetLibrary(0, 4);
        for (int i = 0; i < 100; ++i) {
            {
                CADENCE_STAGE("staged-a", stream);
                Spin<<<8, 64, 0, stream>>>(sink, 64);
            }
            {
                CADENCE_STAGE("staged-b", stream);
                Spin<<<8, 64, 0, stream>>>(sink, 64);
            }
            cadence::Flush();
        }
        const auto snapshot = cadence::Snapshot();
        const std::size_t first = CountOf(snapshot, "staged-a", cadence::ScopeKind::Device);
        const std::size_t second = CountOf(snapshot, "staged-b", cadence::ScopeKind::Device);
        Check(first == 25, "sampleEvery=4 measured 25 of 100 chained iterations");
        // Sampling keeps or drops the complete chain so gaps are not misattributed.
        Check(first == second, "both stages of a chain were kept or dropped together");
        ResetLibrary(0, 1);
    }

    // Events recorded during capture become part of the graph and cannot be
    // queried safely. Scopes must skip captured streams.
    void TestStreamCaptureIsRefusedRatherThanCorrupted(float* sink, cudaStream_t stream) {
        ResetLibrary(0, 1);
        cudaGraph_t graph = nullptr;
        Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) == cudaSuccess,
              "capture began");
        {
            CADENCE_KERNEL("captured-kernel", stream);
            Spin<<<8, 64, 0, stream>>>(sink, 64);
        }
        {
            CADENCE_STAGE("captured-stage", stream);
            Spin<<<8, 64, 0, stream>>>(sink, 64);
        }
        const cudaError_t ended = cudaStreamEndCapture(stream, &graph);
        Check(ended == cudaSuccess, "cudaStreamEndCapture succeeded despite the scopes inside it");
        Check(graph != nullptr, "the application still got its graph back");

        cadence::Flush();
        const auto snapshot = cadence::Snapshot();
        Check(CountOf(snapshot, "captured-kernel", cadence::ScopeKind::Device) == 0 &&
                  CountOf(snapshot, "captured-stage", cadence::ScopeKind::Device) == 0,
              "no measurements were invented for the captured region");
        Check(cadence::CapturedScopeCount() >= 2, "the skipped scopes were counted and reported");
        Check(cadence::FailedRecordCount() == 0,
              "standing down left no broken records behind");

        if (graph) cudaGraphDestroy(graph);
        cudaGetLastError();
        ResetLibrary(0, 1);
    }

    // Flushing a scope recorded during capture invalidates the graph. Verify that
    // instrumentation skips the scope and preserves capture completion.
    void TestFlushInsideCaptureLeavesTheGraphIntact(float* sink, cudaStream_t stream) {
        ResetLibrary(0, 1);
        cudaGraph_t graph = nullptr;
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        {
            CADENCE_KERNEL("captured-then-flushed", stream);
            Spin<<<8, 64, 0, stream>>>(sink, 64);
        }
        cadence::Flush();
        const cudaError_t ended = cudaStreamEndCapture(stream, &graph);
        Check(ended == cudaSuccess, "a flush inside the capture region did not invalidate it");
        Check(graph != nullptr, "the graph survived a flush inside its own capture");
        if (graph) cudaGraphDestroy(graph);
        cudaGetLastError();
        ResetLibrary(0, 1);
    }

    // Bowing out of a captured region must not leak the events the scope had already taken, or a loop that captures every iteration drains the pool.
    void TestCaptureDoesNotLeakEvents(float* sink, cudaStream_t stream) {
        ResetLibrary(0, 1);
        for (int i = 0; i < 20; ++i) {
            CADENCE_KERNEL("warm", stream);
            Spin<<<8, 64, 0, stream>>>(sink, 64);
        }
        cadence::Flush();
        const std::size_t before = cadence::detail::Registry::Instance().LiveEventCount();

        for (int i = 0; i < 50; ++i) {
            cudaGraph_t graph = nullptr;
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            {
                CADENCE_KERNEL("captured-loop", stream);
                Spin<<<8, 64, 0, stream>>>(sink, 64);
            }
            cudaStreamEndCapture(stream, &graph);
            if (graph) cudaGraphDestroy(graph);
            cadence::Flush();
        }
        const std::size_t after = cadence::detail::Registry::Instance().LiveEventCount();
        Check(after == before, "fifty captured iterations created no new events");
        cudaGetLastError();
        ResetLibrary(0, 1);
    }

    // GPU spans must fall inside the host scope that launches and waits for them.
    void TestTraceSpansNestInsideTheirHostScope(float* sink, cudaStream_t stream) {
        cadence::Config config;
        config.reportStream = nullptr;
        config.warmupIterations = 0;
        config.nvtxEnabled = false;
        config.writeOnExit = false;
        config.numWorstIterations = 4;
        config.tracePath = "unused-the-test-reads-the-spans-directly.json";
        cadence::Configure(config);
        cadence::Reset();

        for (int i = 0; i < 20; ++i) {
            {
                CADENCE_SCOPE("iteration");
                {
                    CADENCE_KERNEL("first", stream);
                    Spin<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(sink, SPIN_UNIT);
                }
                {
                    CADENCE_KERNEL("second", stream);
                    Spin<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(sink, SPIN_UNIT);
                }
                cudaStreamSynchronize(stream);
            }
            // Add host delay that would expose a flush-time anchor error.
            BurnHostMicroseconds(1000);
            cadence::Flush();
        }

        const std::vector<cadence::TraceIteration> worst = cadence::WorstIterations();
        Check(!worst.empty(), "the trace retained at least one iteration");

        int checked = 0;
        int contained = 0;
        int ordered = 0;
        for (const cadence::TraceIteration& iteration : worst) {
            const cadence::TraceSpan* loop = nullptr;
            for (const cadence::TraceSpan& span : iteration.spans) {
                if (span.label == "iteration" && span.kind == cadence::ScopeKind::Host) loop = &span;
            }
            if (!loop) continue;
            const std::int64_t loopStart = loop->startNs;
            const std::int64_t loopEnd = loopStart + static_cast<std::int64_t>(loop->durationMs * 1e6);
            const cadence::TraceSpan* first = nullptr;
            const cadence::TraceSpan* second = nullptr;
            for (const cadence::TraceSpan& span : iteration.spans) {
                if (span.kind != cadence::ScopeKind::Device) continue;
                ++checked;
                const std::int64_t spanStart = span.startNs;
                const std::int64_t spanEnd = spanStart + static_cast<std::int64_t>(span.durationMs * 1e6);
                // Allow midpoint timing error while remaining below the injected delay.
                if (spanStart >= loopStart - TRACE_TOLERANCE_NS && spanEnd <= loopEnd + TRACE_TOLERANCE_NS) ++contained;
                if (span.label == "first") first = &span;
                if (span.label == "second") second = &span;
            }
            // Two kernels on one stream cannot overlap, and the second was launched after the first.
            if (first && second && second->startNs >= first->startNs) ++ordered;
        }
        Check(checked > 0, "device spans carried absolute timestamps");
        Check(contained == checked, "every GPU span fell inside the host scope that waited for it");
        Check(ordered == static_cast<int>(worst.size()), "stages on one stream came back in stream order");

        cadence::Config plain;
        plain.reportStream = nullptr;
        plain.writeOnExit = false;
        cadence::Configure(plain);
        cadence::Reset();
    }

    // Rows must retain the device id produced by the single available GPU.
    void TestDeviceRowsCarryTheirDevice(float* sink, cudaStream_t stream) {
        cadence::Config config;
        config.warmupIterations = 0;
        config.reportStream = nullptr;
        cadence::Configure(config);
        cadence::Reset();

        int current = -1;
        cudaGetDevice(&current);
        for (int i = 0; i < 4; ++i) {
            {
                CADENCE_KERNEL("attributed", stream);
                Spin<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(sink, SPIN_UNIT);
            }
            BurnHostMicroseconds(5);
        }
        cudaStreamSynchronize(stream);
        cadence::Flush();

        const auto snapshot = cadence::Snapshot();
        std::size_t deviceRows = 0;
        bool attributed = true;
        for (const cadence::Stats& row : snapshot) {
            if (row.label != "attributed") continue;
            ++deviceRows;
            // Both halves of the label -- the GPU span and the CPU-issue span beside it -- belong to the card the work went to.
            if (row.device != current) attributed = false;
        }
        Check(deviceRows == 2, "one GPU row and one issue row for the label");
        Check(attributed, "both rows name the device that ran the work");

        // One card, so the column that tells cards apart must not appear.
        Check(!cadence::detail::HasMultipleDevices(snapshot), "a single-GPU report hides the device column");
    }

    void TestConcurrentThreadsRecordDeviceWork(float* sink) {
        ResetLibrary(0, 1);
        constexpr int NUM_THREADS_USED = 4;
        constexpr int NUM_ITERATIONS = 100;
        std::vector<std::thread> workers;
        for (int t = 0; t < NUM_THREADS_USED; ++t) {
            workers.emplace_back([&] {
                cudaStream_t own = nullptr;
                cudaStreamCreate(&own);
                for (int i = 0; i < NUM_ITERATIONS; ++i) {
                    CADENCE_KERNEL("threaded", own);
                    Spin<<<8, 64, 0, own>>>(sink, 64);
                }
                cudaStreamSynchronize(own);
                cudaStreamDestroy(own);
            });
        }
        for (std::thread& worker : workers) worker.join();
        cadence::Flush();
        const auto snapshot = cadence::Snapshot();
        Check(CountOf(snapshot, "threaded", cadence::ScopeKind::Device) ==
                  NUM_THREADS_USED * NUM_ITERATIONS,
              "every thread's records survived the flush");
        Check(cadence::FailedRecordCount() == 0, "no records dropped under concurrency");
    }

}  // namespace

int main() {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        std::printf("no CUDA device available; skipping device tests\n");
        return 77;
    }

    float* sink = nullptr;
    if (cudaMalloc(&sink, sizeof(float)) != cudaSuccess) {
        std::printf("could not allocate on the device; skipping device tests\n");
        return 77;
    }
    cudaStream_t stream = nullptr;
    cudaStream_t other = nullptr;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&other);

    TestPairedScopeTracksWork(sink, stream);
    TestChainedStagesTrackWork(sink, stream);
    TestChainingChargesGapsToTheNextStage(sink, stream);
    TestPairedScopeIgnoresGaps(sink, stream);
    TestEventPoolSettles(sink, stream);
    TestChainsArePerStream(sink, stream, other);
    TestSamplingKeepsEveryNth(sink, stream);
    TestStageSamplingThinsWholeChains(sink, stream);
    TestStreamCaptureIsRefusedRatherThanCorrupted(sink, stream);
    TestFlushInsideCaptureLeavesTheGraphIntact(sink, stream);
    TestCaptureDoesNotLeakEvents(sink, stream);
    TestTraceSpansNestInsideTheirHostScope(sink, stream);
    TestDeviceRowsCarryTheirDevice(sink, stream);
    TestConcurrentThreadsRecordDeviceWork(sink);

    cudaStreamDestroy(other);
    cudaStreamDestroy(stream);
    cudaFree(sink);

    std::printf("\n%s\n", failures == 0 ? "all device tests passed" : "device tests FAILED");
    return failures == 0 ? 0 : 1;
}
