// Host-only coverage for report, statistics, and registry behavior.

#include <cadence/cadence.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
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

    // Ensure counted scopes exceed the clock resolution and are not discarded.
    void BurnBriefly() {
        const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(2);
        while (std::chrono::steady_clock::now() < until) {
        }
    }

    // Take ownership so a pointer into a temporary Snapshot remains valid.
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
        config.reportStream = nullptr;  // Keep the test from printing over the test log.
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
        config.reportStream = nullptr;
        cadence::Configure(config);
        cadence::Reset();

        cadence::detail::Registry& registry = cadence::detail::Registry::Instance();
        // The warmup counter persists across flush boundaries.
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
        config.reportStream = nullptr;
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
        config.reportStream = nullptr;
        cadence::Configure(config);
        cadence::Reset();

        { CADENCE_SCOPE("ignored"); }
        cadence::Flush();
        CHECK(cadence::Snapshot().empty());

        config.enabled = true;
        cadence::Configure(config);
    }

    void TestReportOutput() {
        cadence::Config config;
        config.warmupIterations = 1;
        config.reportStream = nullptr;  // Keep the test from printing over the test log.
        cadence::Configure(config);
        cadence::Reset();

        cadence::detail::Registry& registry = cadence::detail::Registry::Instance();
        registry.RecordHost("reportlabel", 9.0);  // Discarded as warmup.
        registry.RecordHost("reportlabel", 1.5);
        registry.RecordHost("reportlabel", 2.5);
        cadence::Flush();

        std::ostringstream out;
        cadence::WriteReport(out);
        const std::string text = out.str();

        // Warmup is report provenance needed to interpret the measurements.
        CHECK(text.find("cadence report") != std::string::npos);
        CHECK(text.find("1 iteration discarded per label") != std::string::npos);
        // The row itself: two kept samples of 1.5 and 2.5 ms, so a 2.00 ms mean rendered in milliseconds.
        CHECK(text.find("reportlabel") != std::string::npos);
        CHECK(text.find("2.00ms") != std::string::npos);
        CHECK(text.find("distribution") != std::string::npos);
    }

    // Choose units independently for each measurement.
    void TestDurationFormatting() {
        CHECK(cadence::detail::FormatDuration(0.0000005, false) == "0.50ns");
        CHECK(cadence::detail::FormatDuration(0.005025, false) == "5.03us");
        CHECK(cadence::detail::FormatDuration(0.0686, false) == "68.6us");
        CHECK(cadence::detail::FormatDuration(0.19, false) == "190us");
        CHECK(cadence::detail::FormatDuration(2.0, false) == "2.00ms");
        CHECK(cadence::detail::FormatDuration(1500.0, false) == "1.50s");
        // The ASCII fallback must replace the two-byte UTF-8 micro sign.
        CHECK(cadence::detail::FormatDuration(0.0686, true) == "68.6\xc2\xb5s");
    }

    // Empty bins must remain visible to preserve distribution shape.
    void TestHistogramRendering() {
        std::array<std::uint32_t, cadence::NUM_HISTOGRAM_BINS> histogram{};
        histogram[0] = 10;
        histogram[11] = 5;
        const std::string ascii = cadence::detail::RenderHistogram(histogram, false);
        CHECK(ascii.size() == cadence::NUM_HISTOGRAM_BINS);
        CHECK(ascii[0] == '#');
        CHECK(ascii[1] == ' ');
        CHECK(ascii[10] == ' ');
        CHECK(ascii[11] != ' ');

        // A label that never varied lands entirely in bin 0 and must not render as a blank row.
        std::array<std::uint32_t, cadence::NUM_HISTOGRAM_BINS> flat{};
        flat[0] = 42;
        CHECK(cadence::detail::RenderHistogram(flat, false)[0] == '#');

        // Nothing recorded renders as nothing at all, rather than as a row of noise.
        std::array<std::uint32_t, cadence::NUM_HISTOGRAM_BINS> empty{};
        CHECK(cadence::detail::RenderHistogram(empty, false).empty());
    }

    // Zero-length host spans are failed clock measurements and must be dropped.
    void TestStalledClockSamplesAreDropped() {
        cadence::Config config;
        config.warmupIterations = 0;
        config.reportStream = nullptr;
        cadence::Configure(config);
        cadence::Reset();

        cadence::detail::Registry& registry = cadence::detail::Registry::Instance();
        registry.RecordHost("stalled", 2.0);
        registry.RecordHost("stalled", 0.0);  // The clock did not move.
        registry.RecordHost("stalled", 4.0);
        cadence::Flush();

        bool found = false;
        const cadence::Stats row = Find("stalled", cadence::ScopeKind::Host, &found);
        CHECK(found);
        CHECK(row.count == 2);
        CHECK_NEAR(row.minMs, 2.0, 1e-9);  // Not zero.
        CHECK_NEAR(row.meanMs, 3.0, 1e-9);
        CHECK(cadence::StalledClockCount() == 1);

        // And the reader is told, rather than silently handed a thinner sample set.
        std::ostringstream out;
        cadence::WriteReport(out);
        CHECK(out.str().find("monotonic clock did not advance") != std::string::npos);
    }

    // Deadline counts and aggregate statistics remain exact after reservoir
    // sampling begins.
    void TestBoundedRetentionKeepsExactAggregates() {
        constexpr std::size_t CAP = 64;
        constexpr int NUM_SAMPLES = 5000;
        cadence::Config config;
        config.warmupIterations = 0;
        config.reportStream = nullptr;
        config.maxSamplesPerLabel = CAP;
        config.budgetMs = 100.0;
        cadence::Configure(config);
        cadence::Reset();

        cadence::detail::Registry& registry = cadence::detail::Registry::Instance();
        // 1.0 through 5000.0 milliseconds, so 4900 of them are over the 100 ms budget.
        for (int i = 1; i <= NUM_SAMPLES; ++i) registry.RecordHost("long-run", static_cast<double>(i));
        cadence::Flush();

        bool found = false;
        const cadence::Stats row = Find("long-run", cadence::ScopeKind::Host, &found);
        CHECK(found);
        CHECK(row.count == static_cast<std::size_t>(NUM_SAMPLES));
        CHECK(row.estimated);
        CHECK_NEAR(row.minMs, 1.0, 1e-6);
        CHECK_NEAR(row.maxMs, 5000.0, 1e-6);
        CHECK_NEAR(row.meanMs, 2500.5, 1e-6);
        CHECK(row.overBudget == 4900);
        // The mean of 1..N is exact, so the standard deviation has a closed form to check it against.
        CHECK_NEAR(row.stddevMs, 1443.5205, 1e-2);

        // Estimated percentiles remain within the expected range.
        CHECK(row.p50Ms > 2000.0 && row.p50Ms < 3000.0);
        CHECK(row.p95Ms > 4300.0 && row.p95Ms < 5000.0);

        // The outlier bin is the reason the distribution column exists, and a reservoir that happened not to retain the slowest sample must not erase it.
        CHECK(row.histogram[cadence::NUM_HISTOGRAM_BINS - 1] > 0);
        CHECK(row.histogram[0] > 0);

        std::ostringstream out;
        cadence::WriteReport(out);
        CHECK(out.str().find("outgrew its sample reservoir") != std::string::npos);

        // A run that fits inside the cap reports nothing as estimated.
        cadence::Reset();
        for (int i = 1; i <= 10; ++i) registry.RecordHost("short-run", static_cast<double>(i));
        cadence::Flush();
        const cadence::Stats small = Find("short-run", cadence::ScopeKind::Host);
        CHECK(!small.estimated);
        CHECK(small.count == 10);
    }

    // The configured reservoir cap bounds retained memory.
    void TestRetentionCapBoundsMemory() {
        constexpr std::size_t CAP = 32;
        cadence::Config config;
        config.warmupIterations = 0;
        config.reportStream = nullptr;
        config.maxSamplesPerLabel = CAP;
        cadence::Configure(config);
        cadence::Reset();

        cadence::detail::Registry& registry = cadence::detail::Registry::Instance();
        for (int i = 0; i < 10000; ++i) registry.RecordHost("capped", 1.0 + (i % 7));
        cadence::Flush();

        const cadence::Stats row = Find("capped", cadence::ScopeKind::Host);
        CHECK(row.count == 10000);

        // Uniform sampling preserves the full observed range.
        CHECK_NEAR(row.minMs, 1.0, 1e-9);
        CHECK_NEAR(row.maxMs, 7.0, 1e-9);
        CHECK_NEAR(row.meanMs, 4.0, 0.05);

        // Zero means retain everything, which is what a short benchmark wants.
        config.maxSamplesPerLabel = 0;
        cadence::Configure(config);
        cadence::Reset();
        for (int i = 0; i < 500; ++i) registry.RecordHost("uncapped", 1.0);
        cadence::Flush();
        CHECK(!Find("uncapped", cadence::ScopeKind::Host).estimated);
    }

    // The summary says how often the loop missed; this says which passes did. Everything between two flushes is one iteration, and the slowest few are kept whole.
    void TestWorstIterationsAreRetained() {
        cadence::Config config;
        config.warmupIterations = 0;
        config.reportStream = nullptr;
        config.numWorstIterations = 3;
        cadence::Configure(config);
        cadence::Reset();

        cadence::detail::Registry& registry = cadence::detail::Registry::Instance();
        // Ten iterations, each a loop scope plus one stage, with iteration i taking i milliseconds.
        for (int i = 1; i <= 10; ++i) {
            registry.RecordHost("iteration", static_cast<double>(i));
            registry.RecordHost("stage", static_cast<double>(i) * 0.5);
            cadence::Flush();
        }

        const std::vector<cadence::TraceIteration> worst = cadence::WorstIterations();
        CHECK(worst.size() == 3);
        // Slowest first, and ranked by the loop scope rather than by the sum of its parts.
        CHECK_NEAR(worst[0].spanMs, 10.0, 1e-6);
        CHECK_NEAR(worst[1].spanMs, 9.0, 1e-6);
        CHECK_NEAR(worst[2].spanMs, 8.0, 1e-6);
        // Each retained iteration keeps every stage that ran inside it.
        CHECK(worst[0].spans.size() == 2);

        bool sawIteration = false;
        bool sawStage = false;
        for (const cadence::TraceSpan& span : worst[0].spans) {
            if (span.label == "iteration") { sawIteration = true; CHECK_NEAR(span.durationMs, 10.0, 1e-6); }
            if (span.label == "stage") { sawStage = true; CHECK_NEAR(span.durationMs, 5.0, 1e-6); }
        }
        CHECK(sawIteration);
        CHECK(sawStage);

        std::ostringstream out;
        cadence::WriteReport(out);
        const std::string text = out.str();
        CHECK(text.find("slowest iterations") != std::string::npos);

        // Zero turns the section off entirely.
        config.numWorstIterations = 0;
        cadence::Configure(config);
        cadence::Reset();
        for (int i = 1; i <= 5; ++i) {
            registry.RecordHost("iteration", static_cast<double>(i));
            cadence::Flush();
        }
        CHECK(cadence::WorstIterations().empty());
        std::ostringstream off;
        cadence::WriteReport(off);
        CHECK(off.str().find("slowest iterations") == std::string::npos);
    }

    // Perfetto reserves thread id 0 for the idle task. Assign distinct nonzero
    // ids so host and device spans render on separate tracks.
    void TestTraceLanesAreDistinctThreads() {
        cadence::TraceIteration iteration;
        iteration.index = 0;
        iteration.spanMs = 0.05;
        iteration.spans.push_back(cadence::TraceSpan{"loop", cadence::ScopeKind::Host, 0.05, 1000, 0});
        iteration.spans.push_back(cadence::TraceSpan{"saxpy", cadence::ScopeKind::Device, 0.04, 2000, 1});
        iteration.spans.push_back(cadence::TraceSpan{"scale", cadence::ScopeKind::Device, 0.02, 3000, 2});

        std::ostringstream out;
        cadence::detail::WriteTraceJson(out, {iteration});
        const std::string json = out.str();

        CHECK(json.find("\"tid\":0") == std::string::npos);
        // Distinct ids, and the metadata naming each lane has to agree with the spans that sit on it.
        CHECK(json.find("\"tid\":1,\"name\":\"thread_name\",\"args\":{\"name\":\"host (CPU)\"}") != std::string::npos);
        CHECK(json.find("\"tid\":2,\"name\":\"thread_name\",\"args\":{\"name\":\"device stream 1\"}") != std::string::npos);
        CHECK(json.find("\"tid\":3,\"name\":\"thread_name\",\"args\":{\"name\":\"device stream 2\"}") != std::string::npos);
        CHECK(json.find("\"tid\":1,\"name\":\"loop\"") != std::string::npos);
        CHECK(json.find("\"tid\":2,\"name\":\"saxpy\"") != std::string::npos);
        CHECK(json.find("\"tid\":3,\"name\":\"scale\"") != std::string::npos);
    }

    // Export retained iterations as a valid, rebased trace.
    void TestTraceJsonShape() {
        cadence::Config config;
        config.warmupIterations = 0;
        config.reportStream = nullptr;
        config.numWorstIterations = 2;
        cadence::Configure(config);
        cadence::Reset();

        cadence::detail::Registry& registry = cadence::detail::Registry::Instance();
        for (int i = 1; i <= 4; ++i) {
            registry.RecordHost("loop\"quoted", static_cast<double>(i));
            cadence::Flush();
        }

        std::ostringstream out;
        cadence::WriteTrace(out);
        const std::string json = out.str();

        CHECK(json.find("\"traceEvents\"") != std::string::npos);
        CHECK(json.find("\"displayTimeUnit\"") != std::string::npos);
        CHECK(json.find("\"process_name\"") != std::string::npos);
        CHECK(json.find("host (CPU)") != std::string::npos);
        // A label carrying a quote has to come back out as valid JSON rather than as a broken file.
        CHECK(json.find("loop\\\"quoted") != std::string::npos);
        // Two iterations retained, one span each.
        std::size_t spans = 0;
        for (std::size_t at = json.find("\"ph\":\"X\""); at != std::string::npos; at = json.find("\"ph\":\"X\"", at + 1)) ++spans;
        CHECK(spans == 2);
        // Durations are microseconds in this format, so a 4 ms span is 4000.
        CHECK(json.find("\"dur\":4000.000") != std::string::npos);
        // Rebased so the earliest span sits at zero rather than at the steady clock's arbitrary origin.
        CHECK(json.find("\"ts\":0.000") != std::string::npos);

        // Braces balance, which is the cheapest proof the file is not truncated.
        int depth = 0;
        bool balanced = true;
        for (char character : json) {
            if (character == '{') ++depth;
            if (character == '}' && --depth < 0) balanced = false;
        }
        CHECK(balanced && depth == 0);
    }

    // Synthesize per-device rows to exercise presentation without CUDA.
    cadence::Stats MakeRow(const char* label, cadence::ScopeKind kind, int device, const std::vector<double>& samples, double budgetMs = 0.0) {
        cadence::Stats row = cadence::detail::ComputeStats(label, kind, samples, 0, budgetMs);
        row.device = device;
        return row;
    }

    std::string RenderTable(const std::vector<cadence::Stats>& rows) {
        std::ostringstream out;
        cadence::detail::WriteStatsTable(out, rows, false, cadence::detail::Theme{});
        return out.str();
    }

    std::string RenderSummary(const std::vector<cadence::Stats>& rows) {
        std::ostringstream out;
        cadence::detail::WriteSummary(out, rows, false, cadence::detail::Theme{});
        return out.str();
    }

    // Summary totals must weight repeated labels by their observations per
    // iteration. Otherwise a graph with many operations under one label is
    // incorrectly classified as host-bound.
    void TestSummaryWeightsLabelsByHowOftenTheyRun() {
        // 10 iterations of a 2.00ms loop body. One kernel runs once per pass at 100us; the other runs ten times per pass at 100us, so it is 1.00ms of the iteration and the naive sum would call it 100us.
        std::vector<cadence::Stats> rows;
        rows.push_back(MakeRow("iteration", cadence::ScopeKind::Host, 0, std::vector<double>(10, 2.0)));
        rows.push_back(MakeRow("once", cadence::ScopeKind::Device, 0, std::vector<double>(10, 0.1)));
        rows.push_back(MakeRow("often", cadence::ScopeKind::Device, 0, std::vector<double>(100, 0.1)));

        const std::string text = RenderSummary(rows);
        // 0.100ms + 10 x 0.100ms = 1.10ms of GPU work per iteration, which is 55% of a 2.00ms pass.
        CHECK(text.find("1.10ms per iteration across 2 labels") != std::string::npos);
        CHECK(text.find("55.0% is GPU work") != std::string::npos);
        CHECK(text.find("900us is launch and synchronization") != std::string::npos);
        // The unweighted answer must not appear anywhere: 200us of device work, wrongly leaving 90% of the pass as overhead.
        CHECK(text.find("200us") == std::string::npos);
        CHECK(text.find("10.0% is GPU work") == std::string::npos);
    }

    // Without a loop scope, report a sum of means instead of a per-iteration rate.
    void TestSummaryWithholdsTheConclusionWithoutALoopScope() {
        std::vector<cadence::Stats> rows;
        rows.push_back(MakeRow("first", cadence::ScopeKind::Device, 0, std::vector<double>(10, 0.1)));
        rows.push_back(MakeRow("second", cadence::ScopeKind::Device, 0, std::vector<double>(100, 0.1)));

        const std::string text = RenderSummary(rows);
        CHECK(text.find("200us across 2 labels, one mean each") != std::string::npos);
        CHECK(text.find("per iteration") == std::string::npos);
        CHECK(text.find("is GPU work") == std::string::npos);

        // Two host-only labels are equally ambiguous: either could be the loop body, so neither is assumed to be.
        rows.push_back(MakeRow("outer", cadence::ScopeKind::Host, 0, std::vector<double>(10, 5.0)));
        rows.push_back(MakeRow("inner", cadence::ScopeKind::Host, 0, std::vector<double>(10, 1.0)));
        CHECK(RenderSummary(rows).find("per iteration") == std::string::npos);
    }

    // Fold and cap graph workloads so the breakdown remains readable.
    void TestWorstIterationBreakdownFoldsAndCaps() {
        cadence::TraceIteration iteration;
        iteration.index = 7;
        iteration.spanMs = 1.0;
        // Ten distinct labels, each running three times, so both the fold and the cap have something to do.
        for (int label = 0; label < 10; ++label) {
            for (int repeat = 0; repeat < 3; ++repeat) {
                iteration.spans.push_back(cadence::TraceSpan{
                    "stage" + std::to_string(label), cadence::ScopeKind::Device, 0.01 * static_cast<double>(label + 1), 0, 1});
            }
        }
        // A host span on the same iteration stays out of the breakdown, as it always has.
        iteration.spans.push_back(cadence::TraceSpan{"loop", cadence::ScopeKind::Host, 1.0, 0, 0});

        std::ostringstream out;
        cadence::detail::WriteWorstIterations(out, {iteration}, false, cadence::detail::Theme{});
        const std::string text = out.str();

        // Repeats fold into one entry carrying their total and their count: 3 x 100us of the widest stage.
        CHECK(text.find("stage9 300us x3") != std::string::npos);
        CHECK(text.find("stage8 270us x3") != std::string::npos);
        // Widest first, so the answer is at the front of the line.
        CHECK(text.find("stage9") < text.find("stage8"));
        // Eight of the ten survive the cap, and the two narrowest are counted rather than printed.
        CHECK(text.find("+2 more") != std::string::npos);
        CHECK(text.find("stage0") == std::string::npos);
        CHECK(text.find("stage1 ") == std::string::npos);
        CHECK(text.find("loop") == std::string::npos);
        // One line for the iteration, and it stays short enough to read.
        CHECK(std::count(text.begin(), text.end(), '\n') == 3);

        // A stage that ran once is still printed without a count, which is the shape every single-stage loop produces.
        cadence::TraceIteration plain;
        plain.spanMs = 0.05;
        plain.spans.push_back(cadence::TraceSpan{"saxpy", cadence::ScopeKind::Device, 0.04, 0, 1});
        std::ostringstream single;
        cadence::detail::WriteWorstIterations(single, {plain}, false, cadence::detail::Theme{});
        CHECK(single.str().find("saxpy 40.0us") != std::string::npos);
        CHECK(single.str().find(" x") == std::string::npos);
    }

    // Rank asynchronous iterations by device work when it exceeds the host span.
    void TestWorstIterationsRankByTheLongerOfHostAndDevice() {
        std::vector<cadence::detail::IterationSpan> queued;
        queued.push_back(cadence::detail::IterationSpan{0, cadence::ScopeKind::Host, 0.008, 0, 0});
        queued.push_back(cadence::detail::IterationSpan{1, cadence::ScopeKind::Device, 3.93, 0, 1});
        CHECK_NEAR(cadence::detail::IterationSpanMs(queued), 3.93, 1e-9);

        // The ordinary shape, where the loop scope does enclose its GPU work, is unchanged: the host span is the larger of the two by construction.
        std::vector<cadence::detail::IterationSpan> enclosed;
        enclosed.push_back(cadence::detail::IterationSpan{0, cadence::ScopeKind::Host, 5.0, 0, 0});
        enclosed.push_back(cadence::detail::IterationSpan{1, cadence::ScopeKind::Device, 2.0, 0, 1});
        enclosed.push_back(cadence::detail::IterationSpan{2, cadence::ScopeKind::Device, 1.0, 0, 1});
        CHECK_NEAR(cadence::detail::IterationSpanMs(enclosed), 5.0, 1e-9);

        // Device-only, which has no host span to prefer in the first place.
        std::vector<cadence::detail::IterationSpan> deviceOnly;
        deviceOnly.push_back(cadence::detail::IterationSpan{0, cadence::ScopeKind::Device, 2.0, 0, 1});
        CHECK_NEAR(cadence::detail::IterationSpanMs(deviceOnly), 2.0, 1e-9);
    }

    // Each GPU keeps an independent row so device differences remain visible.
    void TestDeviceSamplesKeyByDevice() {
        cadence::detail::LabelSamples samples;
        cadence::detail::SlotFor(samples, 7, 1).gpu.Add(2.0, 0.0, 0);
        cadence::detail::SlotFor(samples, 7, 0).gpu.Add(1.0, 0.0, 0);
        cadence::detail::SlotFor(samples, 7, 1).gpu.Add(4.0, 0.0, 0);
        cadence::detail::SlotFor(samples, 7, 0).issue.Add(0.5, 0.0, 0);

        // Two slots, not three and not one, and ordered by device id however they arrived.
        CHECK(samples.devices.size() == 2);
        CHECK(samples.devices[0].device == 0);
        CHECK(samples.devices[1].device == 1);
        CHECK(samples.devices[0].gpu.count == 1);
        CHECK(samples.devices[1].gpu.count == 2);
        CHECK_NEAR(samples.devices[0].gpu.mean, 1.0, 1e-9);
        CHECK_NEAR(samples.devices[1].gpu.mean, 3.0, 1e-9);
        // The issue side is keyed the same way and stays on its own device.
        CHECK(samples.devices[0].issue.count == 1);
        CHECK(samples.devices[1].issue.count == 0);
        // Distinct seeds, so two cards do not thin their reservoirs in lockstep.
        CHECK(samples.devices[0].gpu.rngState != samples.devices[1].gpu.rngState);
    }

    // The device column costs a reader nothing to ignore but everything to misread, so it appears only when there is a second GPU to tell apart.
    void TestDeviceColumnAppearsOnlyWithSeveralDevices() {
        const std::vector<double> fast = {1.0, 1.1, 1.2};
        const std::vector<double> slow = {4.0, 4.1, 4.2};

        std::vector<cadence::Stats> single = {
            MakeRow("saxpy", cadence::ScopeKind::Device, 0, fast),
            MakeRow("iteration", cadence::ScopeKind::Host, -1, slow),
        };
        const std::string one = RenderTable(single);
        CHECK(one.find(" dev ") == std::string::npos);

        std::vector<cadence::Stats> several = single;
        several.push_back(MakeRow("saxpy", cadence::ScopeKind::Device, 1, slow));
        const std::string two = RenderTable(several);
        CHECK(two.find("scope   dev  n") != std::string::npos);
        CHECK(two.find("saxpy      device    0  3") != std::string::npos);
        CHECK(two.find("saxpy      device    1  3") != std::string::npos);
        // A plain host span belongs to no GPU, so its cell stays blank rather than claiming device 0.
        CHECK(two.find("iteration  host         3") != std::string::npos);
    }

    // With one label on several GPUs every one of its rows carries the deadline, and there is one line to spend. A verdict that reads "met" while another card missed is the most misleading thing this report could print.
    void TestBudgetLineNamesTheWorstDevice() {
        std::vector<cadence::Stats> rows = {
            MakeRow("infer", cadence::ScopeKind::Device, 0, {1.0, 1.1, 1.2}, 2.0),
            MakeRow("infer", cadence::ScopeKind::Device, 1, {1.0, 3.0, 4.0}, 2.0),
        };
        std::ostringstream out;
        cadence::detail::WriteBudget(out, rows, false, cadence::detail::Theme{});
        const std::string text = out.str();
        CHECK(text.find("MISSED") != std::string::npos);
        CHECK(text.find("on device 1, the worst of 2") != std::string::npos);
        CHECK(text.find("1/3 iterations inside budget") != std::string::npos);

        // Sorting the misses later must not change the verdict; the row is chosen, not taken last.
        std::swap(rows[0], rows[1]);
        std::ostringstream reordered;
        cadence::detail::WriteBudget(reordered, rows, false, cadence::detail::Theme{});
        CHECK(reordered.str() == text);
    }

    // A verdict of MISSED beside a share of 100.0% is the confusion the deadline line exists to prevent, and a long run is exactly where rounding produces it.
    void TestNearPerfectShareDoesNotRoundToWhole() {
        std::vector<double> samples(3157, 1.0);
        samples[0] = 3.0;  // One miss in 3157 is 99.968%, which rounds up.
        std::vector<cadence::Stats> rows = {MakeRow("loop", cadence::ScopeKind::Host, -1, samples, 2.0)};
        std::ostringstream out;
        cadence::detail::WriteBudget(out, rows, false, cadence::detail::Theme{});
        const std::string text = out.str();
        CHECK(text.find("MISSED") != std::string::npos);
        CHECK(text.find("3156/3157 iterations inside budget (99.9%)") != std::string::npos);
        CHECK(text.find("100.0%") == std::string::npos);

        // And a run that genuinely held every iteration still reads as a whole 100.0%.
        std::vector<cadence::Stats> clean = {MakeRow("loop", cadence::ScopeKind::Host, -1, std::vector<double>(64, 1.0), 2.0)};
        std::ostringstream perfect;
        cadence::detail::WriteBudget(perfect, clean, false, cadence::detail::Theme{});
        CHECK(perfect.str().find("64/64 iterations inside budget (100.0%)") != std::string::npos);
    }

    // Two GPUs run at the same time, so adding their spans together produces a duration no iteration ever took -- one that would grow with every card added while the loop got faster.
    void TestSummaryTotalsPerDevice() {
        std::vector<cadence::Stats> rows = {
            MakeRow("saxpy", cadence::ScopeKind::Device, 0, {1.0, 1.0, 1.0}),
            MakeRow("scale", cadence::ScopeKind::Device, 0, {2.0, 2.0, 2.0}),
            MakeRow("saxpy", cadence::ScopeKind::Device, 1, {4.0, 4.0, 4.0}),
            MakeRow("iteration", cadence::ScopeKind::Host, -1, {20.0, 20.0, 20.0}),
        };
        std::ostringstream out;
        cadence::detail::WriteSummary(out, rows, false, cadence::detail::Theme{});
        const std::string text = out.str();
        CHECK(text.find("3.00ms per iteration across 2 labels on device 0") != std::string::npos);
        CHECK(text.find("4.00ms per iteration across 1 label on device 1") != std::string::npos);
        CHECK(text.find("7.00ms") == std::string::npos);
        // The remainder is launch and synchronization only while there is one GPU to subtract, so with several the line is withheld rather than guessed at.
        CHECK(text.find("launch and synchronization") == std::string::npos);

        // With a single device it reads exactly as it always did.
        rows.erase(rows.begin() + 2);
        std::ostringstream one;
        cadence::detail::WriteSummary(one, rows, false, cadence::detail::Theme{});
        const std::string single = one.str();
        CHECK(single.find("3.00ms per iteration across 2 labels") != std::string::npos);
        CHECK(single.find("on device") == std::string::npos);
        CHECK(single.find("launch and synchronization") != std::string::npos);
    }

    void TestBudgetCountsMisses() {
        cadence::Config config;
        config.warmupIterations = 0;
        config.reportStream = nullptr;
        config.budgetMs = 10.0;
        cadence::Configure(config);
        cadence::Reset();

        cadence::detail::Registry& registry = cadence::detail::Registry::Instance();
        for (int i = 0; i < 8; ++i) registry.RecordHost("loop", 5.0);
        registry.RecordHost("loop", 12.0);  // Missed.
        registry.RecordHost("loop", 30.0);  // Missed.
        cadence::Flush();

        bool found = false;
        const cadence::Stats row = Find("loop", cadence::ScopeKind::Host, &found);
        CHECK(found);
        CHECK(row.count == 10);
        CHECK_NEAR(row.budgetMs, 10.0, 1e-9);
        CHECK(row.overBudget == 2);

        std::ostringstream out;
        cadence::WriteReport(out);
        const std::string text = out.str();
        CHECK(text.find("MISSED") != std::string::npos);
        CHECK(text.find("8/10 iterations inside budget") != std::string::npos);

        // A sample exactly on the deadline is inside it, not over it.
        cadence::Reset();
        for (int i = 0; i < 4; ++i) registry.RecordHost("loop", 10.0);
        cadence::Flush();
        const cadence::Stats exact = Find("loop", cadence::ScopeKind::Host, &found);
        CHECK(found);
        CHECK(exact.overBudget == 0);
    }

    // With no label named the budget has to pick one row on its own, and picking the wrong one is worse than picking none.
    void TestBudgetTargetSelection() {
        cadence::Config config;
        config.warmupIterations = 0;
        config.reportStream = nullptr;
        config.budgetMs = 10.0;
        cadence::Configure(config);
        cadence::Reset();

        cadence::detail::Registry& registry = cadence::detail::Registry::Instance();
        registry.RecordHost("alpha", 1.0);
        registry.RecordHost("beta", 2.0);
        cadence::Flush();

        // Two host-only labels: ambiguous, so no row carries the budget.
        bool found = false;
        CHECK(Find("alpha", cadence::ScopeKind::Host, &found).budgetMs == 0.0);
        CHECK(Find("beta", cadence::ScopeKind::Host, &found).budgetMs == 0.0);

        // Naming one resolves it.
        config.budgetLabel = "beta";
        cadence::Configure(config);
        CHECK(Find("alpha", cadence::ScopeKind::Host, &found).budgetMs == 0.0);
        CHECK_NEAR(Find("beta", cadence::ScopeKind::Host, &found).budgetMs, 10.0, 1e-9);

        // A label that does not exist holds nothing, rather than falling back to a guess.
        config.budgetLabel = "nonexistent";
        cadence::Configure(config);
        CHECK(Find("beta", cadence::ScopeKind::Host, &found).budgetMs == 0.0);
    }

    // Samples land in the bin their magnitude earns, which is what makes the column readable at a glance.
    void TestHistogramBinning() {
        cadence::Config config;
        config.warmupIterations = 0;
        config.reportStream = nullptr;
        cadence::Configure(config);
        cadence::Reset();

        cadence::detail::Registry& registry = cadence::detail::Registry::Instance();
        for (int i = 0; i < 20; ++i) registry.RecordHost("bimodal", 1.0);
        for (int i = 0; i < 5; ++i) registry.RecordHost("bimodal", 10.0);
        cadence::Flush();

        bool found = false;
        const cadence::Stats row = Find("bimodal", cadence::ScopeKind::Host, &found);
        CHECK(found);
        CHECK(row.histogram[0] == 20);                              // The 1.0 ms cluster.
        CHECK(row.histogram[cadence::NUM_HISTOGRAM_BINS - 1] == 5);  // The 10.0 ms cluster.
        std::uint32_t total = 0;
        for (std::uint32_t count : row.histogram) total += count;
        CHECK(total == 25);
    }

    void TestResetClearsEverything() {
        cadence::Config config;
        config.warmupIterations = 0;
        config.reportStream = nullptr;
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
        config.reportStream = nullptr;
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
        config.reportStream = nullptr;
        cadence::Configure(config);
        cadence::Reset();

        // Each scope brackets real work. An empty scope can measure zero nanoseconds, and a zero-length span is discarded as a stalled clock, which would make the count below depend on timing rather than on interning.
        const std::string built = std::string("mer") + "ged";
        {
            CADENCE_SCOPE("merged");
            BurnBriefly();
        }
        {
            cadence::ScopedHost fromRuntimeString(built.c_str());
            BurnBriefly();
        }
        {
            cadence::ScopedHost fromOtherLiteral("merged");
            BurnBriefly();
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
        config.reportStream = nullptr;
        config.sampleEvery = 5;
        cadence::Configure(config);
        cadence::Reset();

        for (int i = 0; i < 50; ++i) {
            CADENCE_SCOPE("sampled-host");
            BurnBriefly();
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
        {"report output", TestReportOutput},
        {"duration formatting", TestDurationFormatting},
        {"histogram rendering", TestHistogramRendering},
        {"histogram binning", TestHistogramBinning},
        {"stalled clock samples dropped", TestStalledClockSamplesAreDropped},
        {"bounded retention keeps exact aggregates", TestBoundedRetentionKeepsExactAggregates},
        {"retention cap bounds memory", TestRetentionCapBoundsMemory},
        {"worst iterations retained", TestWorstIterationsAreRetained},
        {"summary weights labels by rate", TestSummaryWeightsLabelsByHowOftenTheyRun},
        {"summary withholds without a loop scope", TestSummaryWithholdsTheConclusionWithoutALoopScope},
        {"worst iteration breakdown folds and caps", TestWorstIterationBreakdownFoldsAndCaps},
        {"worst iterations rank by the longer span", TestWorstIterationsRankByTheLongerOfHostAndDevice},
        {"trace lanes are distinct threads", TestTraceLanesAreDistinctThreads},
        {"trace json shape", TestTraceJsonShape},
        {"device samples key by device", TestDeviceSamplesKeyByDevice},
        {"device column appears only with several devices", TestDeviceColumnAppearsOnlyWithSeveralDevices},
        {"budget line names the worst device", TestBudgetLineNamesTheWorstDevice},
        {"summary totals per device", TestSummaryTotalsPerDevice},
        {"near-perfect share does not round to whole", TestNearPerfectShareDoesNotRoundToWhole},
        {"budget counts misses", TestBudgetCountsMisses},
        {"budget target selection", TestBudgetTargetSelection},
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
