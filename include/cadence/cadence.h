// cadence: a header-only, drop-in CUDA timing library for real-time loops.
//
// Include this one header. Everything else lives under cadence/detail/.
//
//   #include <cadence/cadence.h>
//
//   for (int step = 0; step < steps; ++step) {
//     { CADENCE_KERNEL("gemm", stream); Gemm<<<grid, block, 0, stream>>>(...); }
//     CADENCE_FLUSH();            // once per loop iteration, not per kernel
//   }
//   CADENCE_REPORT();             // prints the report

#pragma once  // ensure the header is only pulled in one time

#include <cstdlib>
#include <fstream>
#include <ostream>
#include <string>
#include <vector>

#include "cadence/detail/config.h"
#include "cadence/detail/labels.h"
#include "cadence/detail/platform.h"
#include "cadence/detail/registry.h"
#include "cadence/detail/report.h"
#include "cadence/detail/scopes.h"
#include "cadence/detail/stats.h"

namespace cadence {
    // Apply a configuration. Environment variables still win over whatever is set here; see detail/config.h for the list.
    inline void Configure(const Config& config) { detail::Registry::Instance().Configure(config); }
    inline Config GetConfig() { return detail::Registry::Instance().GetConfig(); }

    // Consume pending records: wait on the recorded stop events, compute elapsed times, discard warmup, fold into per-label statistics.
    // This synchronizes. Call it at a loop or frame boundary
    inline void Flush() { detail::Registry::Instance().Flush(); }

    // Per-label statistics for everything flushed so far.
    inline std::vector<Stats> Snapshot() { return detail::Registry::Instance().Snapshot(); }

    // Drop all accumulated statistics. Warmup counters reset too, so the next N observations per label are discarded again.
    inline void Reset() { detail::Registry::Instance().Reset(); }

    inline RunInfo QueryRunInfo() { return detail::QueryRunInfo(); }

    // Records dropped because event creation or an elapsed-time query failed.
    inline std::size_t FailedRecordCount() { return detail::Registry::Instance().FailedRecordCount(); }

    // Host spans dropped because the monotonic clock did not advance across them. Nonzero says the machine's clock is unreliable under load, not that the measured code was fast.
    inline std::size_t StalledClockCount() { return detail::Registry::Instance().StalledClockCount(); }

    // Render the report to an already-open stream. Does not flush first.
    inline void WriteReport(std::ostream& out) { detail::Registry::Instance().WriteTo(out); }

    // Returns false if the file could not be opened.
    inline bool WriteReport(const std::string& path) {
        std::ofstream out(path);
        if (!out) return false;
        WriteReport(out);
        return out.good();
    }

    // Flush, then print the report to the configured stream and, if outputPath is set, write the same text there. The one call most applications need at shutdown; it also cancels the exit-time fallback.
    inline bool Report() {
        Flush();
        const Config config = GetConfig();
        detail::Registry::Instance().MarkReported();
        if (config.reportStream) WriteReport(*config.reportStream);
        if (config.outputPath.empty()) return true;
        return WriteReport(config.outputPath);
    }

}  // namespace cadence

// Prefer these over the classes: they are what -DCADENCE_DISABLE removes, and they are the only form that resolves a label once per call site rather than once per execution.
#if CADENCE_ENABLED && CADENCE_HAS_CUDA
// CADENCE_KERNEL("label")            -- times work on the default stream
// CADENCE_KERNEL("label", stream)    -- times work on `stream`
#define CADENCE_KERNEL(...) CADENCE_DETAIL_KERNEL_IMPL(__VA_ARGS__, 0, 0)
#define CADENCE_DETAIL_KERNEL_IMPL(label, stream, ...)                                              \
    ::cadence::ScopedKernel CADENCE_DETAIL_UNIQUE(cadenceKernelScope_)(CADENCE_DETAIL_LABEL(label), \
                                                                       stream)

// CADENCE_STAGE("label", stream) -- one event per stage instead of two, at the price of charging any gap on the stream to the stage that follows it. See ScopedStage in detail/scopes.h before reaching for it.
#define CADENCE_STAGE(...) CADENCE_DETAIL_STAGE_IMPL(__VA_ARGS__, 0, 0)
#define CADENCE_DETAIL_STAGE_IMPL(label, stream, ...)                                             \
    ::cadence::ScopedStage CADENCE_DETAIL_UNIQUE(cadenceStageScope_)(CADENCE_DETAIL_LABEL(label), \
                                                                     stream)
#else
#define CADENCE_KERNEL(...) ((void)0)
#define CADENCE_STAGE(...) ((void)0)
#endif

#if CADENCE_ENABLED
#define CADENCE_SCOPE(label) \
    ::cadence::ScopedHost CADENCE_DETAIL_UNIQUE(cadenceHostScope_)(CADENCE_DETAIL_LABEL(label))
#define CADENCE_FLUSH() ::cadence::Flush()
#define CADENCE_REPORT() ::cadence::Report()
#define CADENCE_CONFIGURE(config) ::cadence::Configure(config)
#else
#define CADENCE_SCOPE(label) ((void)0)
#define CADENCE_FLUSH() ((void)0)
#define CADENCE_REPORT() ((void)0)
#define CADENCE_CONFIGURE(config) ((void)0)
#endif
