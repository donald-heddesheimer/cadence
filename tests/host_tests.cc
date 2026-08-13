// Host-only tests for cadence. No CUDA, no GPU, no driver required: these cover the logic that decides whether the reported numbers are correct.

#include <cadence/cadence.h>

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

    int gFailures = 0;

    void Check(bool condition, const char* what, int line) {
        if (!condition) {
            std::printf("  FAIL (line %d): %s\n", line, what);
            ++gFailures;
        }
    }

    void CheckNear(double actual, double expected, double tolerance, const char* what, int line) {
        const double difference = actual > expected ? actual - expected : expected - actual;
        if (difference > tolerance) {
            std::printf("  FAIL (line %d): %s -- got %f, expected %f\n", line, what, actual, expected);
            ++gFailures;
        }
    }

#define CHECK(cond) Check((cond), #cond, __LINE__)
#define CHECK_NEAR(actual, expected, tol) CheckNear((actual), (expected), (tol), #actual, __LINE__)

    // Finds one row of a Snapshot() by label and scope. Takes the snapshot itself rather than a reference into one. Returning a pointer into a vector the caller passed as a temporary is a use-after-free waiting for the first person who writes Find(cadence::Snapshot(), ...).
    cadence::Stats Find(const std::string& label, cadence::ScopeKind kind, bool* found = nullptr) {
        const std::vector<cadence::Stats> rows = cadence::Snapshot();
        for (const cadence::Stats& row : rows) {
            if (row.label == label && row.kind == kind) {
                if (found) *found = true;
                return row;
            }
        }
        if (found) *found = false;
        return cadence::Stats{};
    }

    void TestPercentileNearestRank() {
        const std::vector<double> sorted = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        CHECK_NEAR(cadence::detail::PercentileSorted(sorted, 0.50), 5.0, 1e-9);
        CHECK_NEAR(cadence::detail::PercentileSorted(sorted, 0.95), 10.0, 1e-9);
        CHECK_NEAR(cadence::detail::PercentileSorted(sorted, 0.0), 1.0, 1e-9);
        CHECK_NEAR(cadence::detail::PercentileSorted(sorted, 1.0), 10.0, 1e-9);
        CHECK_NEAR(cadence::detail::PercentileSorted({}, 0.5), 0.0, 1e-9);
    }

    void TestStatistics() {
        const std::vector<double> samples = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
        const cadence::Stats stats =
            cadence::detail::ComputeStats("x", cadence::ScopeKind::Host, samples, 0);
        CHECK(stats.count == 8);
        CHECK_NEAR(stats.meanMs, 5.0, 1e-9);
        CHECK_NEAR(stats.minMs, 2.0, 1e-9);
        CHECK_NEAR(stats.maxMs, 9.0, 1e-9);
        CHECK_NEAR(stats.jitterMs, 7.0, 1e-9);
        // Sample standard deviation (n-1), not the population form.
        CHECK_NEAR(stats.stddevMs, 2.13809, 1e-5);

        const cadence::Stats single =
            cadence::detail::ComputeStats("x", cadence::ScopeKind::Host, {3.0}, 0);
        CHECK(single.count == 1);
        CHECK_NEAR(single.stddevMs, 0.0, 1e-9);
        CHECK_NEAR(single.jitterMs, 0.0, 1e-9);

        const cadence::Stats empty = cadence::detail::ComputeStats("x", cadence::ScopeKind::Host, {}, 4);
        CHECK(empty.count == 0);
        CHECK(empty.discarded == 4);
    }

    void TestWarmupDiscard() {
        cadence::Config config;
        config.warmupIterations = 3;
        config.outputPath.clear();  // Keep the test from littering the build dir.
        cadence::Configure(config);
        cadence::Reset();

        cadence::detail::Registry& registry = cadence::detail::Registry::Instance();
        // The first three observations are the expensive ones (context creation, JIT, library autotuning); they must not reach the statistics.
        for (int i = 0; i < 3; ++i) registry.RecordHost("warm", 100.0);
        for (int i = 0; i < 5; ++i) registry.RecordHost("warm", 1.0);
        cadence::Flush();

        bool found = false;
        const cadence::Stats warm = Find("warm", cadence::ScopeKind::Host, &found);
        CHECK(found);
        {
            CHECK(warm.count == 5);
            CHECK(warm.discarded == 3);
            CHECK_NEAR(warm.meanMs, 1.0, 1e-9);
            CHECK_NEAR(warm.maxMs, 1.0, 1e-9);  // No 100.0 outlier leaked through.
        }
    }

    void TestWarmupPersistsAcrossFlushes() {
        cadence::Config config;
        config.warmupIterations = 2;
        config.outputPath.clear();
        cadence::Configure(config);
        cadence::Reset();

        cadence::detail::Registry& registry = cadence::detail::Registry::Instance();
        // One observation per flush: the counter has to survive the flush boundary, which is the whole point of keeping it in the registry.
        for (int i = 0; i < 4; ++i) {
            registry.RecordHost("perloop", 2.0);
            cadence::Flush();
        }

        bool found = false;
        const cadence::Stats row = Find("perloop", cadence::ScopeKind::Host, &found);
        CHECK(found);
        {
            CHECK(row.count == 2);
            CHECK(row.discarded == 2);
        }
    }

    void TestScopedHostMeasuresRealTime() {
        cadence::Config config;
        config.warmupIterations = 0;
        config.outputPath.clear();
        cadence::Configure(config);
        cadence::Reset();

        {
            CADENCE_SCOPE("sleep");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        cadence::Flush();

        bool found = false;
        const cadence::Stats row = Find("sleep", cadence::ScopeKind::Host, &found);
        CHECK(found);
        {
            CHECK(row.count == 1);
            // Generous upper bound: this runs on shared CI hardware.
            CHECK(row.meanMs >= 19.0);
            CHECK(row.meanMs < 500.0);
        }
    }

    void TestRuntimeDisableStopsCollection() {
        cadence::Config config;
        config.enabled = false;
        config.outputPath.clear();
        cadence::Configure(config);
        cadence::Reset();

        { CADENCE_SCOPE("ignored"); }
        cadence::Flush();
        CHECK(cadence::Snapshot().empty());

        config.enabled = true;
        cadence::Configure(config);
    }

    void TestCsvOutput() {
        cadence::Config config;
        config.warmupIterations = 1;
        config.outputPath.clear();
        cadence::Configure(config);
        cadence::Reset();

        cadence::detail::Registry& registry = cadence::detail::Registry::Instance();
        registry.RecordHost("csvlabel", 9.0);  // Discarded as warmup.
        registry.RecordHost("csvlabel", 1.5);
        registry.RecordHost("csvlabel", 2.5);
        cadence::Flush();

        std::ostringstream out;
        cadence::WriteCsv(out);
        const std::string text = out.str();

        // Provenance header: warmup and clock state must be visible in the artifact, so nobody quotes a throttled run as gospel.
        CHECK(text.find("# cadence report") != std::string::npos);
        CHECK(text.find("warmup_iterations_discarded_per_label: 1") != std::string::npos);
        CHECK(text.find("clock_state:") != std::string::npos);
        CHECK(text.find("label,scope,count,warmup_discarded,mean_ms") != std::string::npos);
        CHECK(text.find("csvlabel,host,2,1,2.000000") != std::string::npos);
    }

    void TestResetClearsEverything() {
        cadence::Config config;
        config.warmupIterations = 0;
        config.outputPath.clear();
        cadence::Configure(config);
        cadence::Reset();

        cadence::detail::Registry::Instance().RecordHost("gone", 1.0);
        cadence::Flush();
        CHECK(!cadence::Snapshot().empty());
        cadence::Reset();
        CHECK(cadence::Snapshot().empty());
    }

    void TestEnvironmentOverridesWinOverStruct() {
        cadence::Config config;
        config.warmupIterations = 7;
        config.nvtxEnabled = true;

        setenv("CADENCE_WARMUP", "11", 1);
        setenv("CADENCE_NVTX", "off", 1);
        cadence::detail::ApplyEnvironmentOverrides(config);
        CHECK(config.warmupIterations == 11);
        CHECK(config.nvtxEnabled == false);
        unsetenv("CADENCE_WARMUP");
        unsetenv("CADENCE_NVTX");

        // An unset variable leaves the struct value alone.
        cadence::Config untouched;
        untouched.warmupIterations = 5;
        cadence::detail::ApplyEnvironmentOverrides(untouched);
        CHECK(untouched.warmupIterations == 5);
    }

    void TestThreadSafeRecording() {
        cadence::Config config;
        config.warmupIterations = 0;
        config.outputPath.clear();
        cadence::Configure(config);
        cadence::Reset();

        constexpr int NUM_THREADS = 4;
        constexpr int NUM_SAMPLES_PER_THREAD = 250;
        std::vector<std::thread> workers;
        for (int t = 0; t < NUM_THREADS; ++t) {
            workers.emplace_back([] {
                for (int i = 0; i < NUM_SAMPLES_PER_THREAD; ++i) {
                    cadence::detail::Registry::Instance().RecordHost("shared", 1.0);
                }
            });
        }
        for (std::thread& worker : workers) worker.join();
        cadence::Flush();

        bool found = false;
        const cadence::Stats row = Find("shared", cadence::ScopeKind::Host, &found);
        CHECK(found);
        CHECK(row.count == NUM_THREADS * NUM_SAMPLES_PER_THREAD);
    }

    // Interning has to key on the characters, not on the pointer: the same literal in two translation units is two addresses, and a label built at runtime is a third. All of them are one label.
    void TestLabelInterningMergesByContent() {
        cadence::Config config;
        config.warmupIterations = 0;
        config.outputPath.clear();
        cadence::Configure(config);
        cadence::Reset();

        const std::string built = std::string("mer") + "ged";
        {
            CADENCE_SCOPE("merged");
            cadence::ScopedHost fromRuntimeString(built.c_str());
            cadence::ScopedHost fromOtherLiteral("merged");
        }
        cadence::Flush();

        bool found = false;
        const cadence::Stats row = Find("merged", cadence::ScopeKind::Host, &found);
        CHECK(found);
        CHECK(row.count == 3);

        // And distinct labels must stay distinct.
        const std::vector<cadence::Stats> rows = cadence::Snapshot();
        std::size_t matching = 0;
        for (const cadence::Stats& candidate : rows) {
            if (candidate.label == "merged") ++matching;
        }
        CHECK(matching == 1);
    }

    // A label interned once must survive the table growing under it. The handle a call site caches points into the table, so a container that moved its elements would leave every earlier call site holding a dangling name.
    void TestInternedHandlesSurviveTableGrowth() {
        const cadence::detail::LabelHandle first =
            cadence::detail::LabelTable::Instance().Intern("survivor");
        const std::string firstName = first.name;
        for (int i = 0; i < 512; ++i) {
            cadence::detail::LabelTable::Instance().Intern(("filler-" + std::to_string(i)).c_str());
        }
        CHECK(std::string(first.name) == firstName);
        CHECK(std::string(first.name) == "survivor");
        const cadence::detail::LabelHandle again =
            cadence::detail::LabelTable::Instance().Intern("survivor");
        CHECK(again.id == first.id);
        CHECK(again.observations == first.observations);
    }

    void TestSamplingKeepsEveryNth() {
        cadence::Config config;
        config.warmupIterations = 0;
        config.outputPath.clear();
        config.sampleEvery = 5;
        cadence::Configure(config);
        cadence::Reset();

        for (int i = 0; i < 50; ++i) {
            CADENCE_SCOPE("sampled-host");
        }
        cadence::Flush();

        bool found = false;
        const cadence::Stats row = Find("sampled-host", cadence::ScopeKind::Host, &found);
        CHECK(found);
        CHECK(row.count == 10);

        config.sampleEvery = 1;
        cadence::Configure(config);
        cadence::Reset();
    }

    struct TestCase {
        const char* name;
        void (*run)();
    };

}  // namespace

int main() {
    const TestCase tests[] = {
        {"percentile nearest rank", TestPercentileNearestRank},
        {"statistics", TestStatistics},
        {"warmup discard", TestWarmupDiscard},
        {"warmup persists across flushes", TestWarmupPersistsAcrossFlushes},
        {"scoped host timer", TestScopedHostMeasuresRealTime},
        {"runtime disable", TestRuntimeDisableStopsCollection},
        {"csv output", TestCsvOutput},
        {"reset", TestResetClearsEverything},
        {"environment overrides", TestEnvironmentOverridesWinOverStruct},
        {"thread-safe recording", TestThreadSafeRecording},
        {"label interning merges by content", TestLabelInterningMergesByContent},
        {"interned handles survive table growth", TestInternedHandlesSurviveTableGrowth},
        {"sampling keeps every nth", TestSamplingKeepsEveryNth},
    };

    for (const TestCase& test : tests) {
        const int before = gFailures;
        test.run();
        std::printf("%s %s\n", gFailures == before ? "[ ok ]" : "[FAIL]", test.name);
    }

    if (gFailures == 0) {
        std::printf("\nall tests passed\n");
        return 0;
    }
    std::printf("\n%d check(s) failed\n", gFailures);
    return 1;
}
