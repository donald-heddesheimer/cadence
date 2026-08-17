// cadence: in-process CPU and CUDA timing for iterative workloads.
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

#pragma once

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
#include "cadence/detail/trace.h"

namespace cadence {
    // Apply configuration. Environment variables take precedence.
    inline void Configure(const Config& config) { detail::Registry::Instance().Configure(config); }
    inline Config GetConfig() { return detail::Registry::Instance().GetConfig(); }

    // Resolve pending records and update statistics. This synchronizes, so call
    // it at a loop or frame boundary.
    inline void Flush() { detail::Registry::Instance().Flush(); }

    // Per-label statistics for everything flushed so far.
    inline std::vector<Stats> Snapshot() { return detail::Registry::Instance().Snapshot(); }

    // Clear statistics and restart warmup counters.
    inline void Reset() { detail::Registry::Instance().Reset(); }

    inline RunInfo QueryRunInfo() { return detail::QueryRunInfo(); }

    // Records dropped because event creation or an elapsed-time query failed.
    inline std::size_t FailedRecordCount() { return detail::Registry::Instance().FailedRecordCount(); }

    // Host spans dropped because the monotonic clock did not advance.
    inline std::size_t StalledClockCount() { return detail::Registry::Instance().StalledClockCount(); }

    // Scopes skipped during CUDA graph capture. Always zero without CUDA.
    inline std::size_t CapturedScopeCount() {
#if CADENCE_HAS_CUDA
        return detail::Registry::Instance().CapturedScopeCount();
#else
        return 0;
#endif
    }

    // Retained slow iterations, ordered slowest first. An iteration contains all
    // records between two flushes.
    inline std::vector<TraceIteration> WorstIterations() { return detail::Registry::Instance().WorstIterations(); }

    // Render the report to an already-open stream. Does not flush first.
    inline void WriteReport(std::ostream& out) { detail::Registry::Instance().WriteTo(out); }

    // Write retained iterations as Chrome Trace Event JSON. Absolute GPU
    // placement is available when Config::tracePath was set before recording.
    inline void WriteTrace(std::ostream& out) { detail::Registry::Instance().WriteTraceTo(out); }

    // Returns false if the file could not be opened.
    inline bool WriteTrace(const std::string& path) {
        std::ofstream out(path);
        if (!out) return false;
        WriteTrace(out);
        return out.good();
    }

    // Returns false if the file could not be opened.
    inline bool WriteReport(const std::string& path) {
        std::ofstream out(path);
        if (!out) return false;
        WriteReport(out);
        return out.good();
    }

    // Flush and write configured report and trace outputs. Calling Report()
    // disables the exit-time fallback.
    inline bool Report() {
        Flush();
        const Config config = GetConfig();
        detail::Registry::Instance().MarkReported();
        if (config.reportStream) WriteReport(*config.reportStream);
        bool ok = true;
        if (!config.tracePath.empty()) ok = WriteTrace(config.tracePath);
        if (config.outputPath.empty()) return ok;
        return WriteReport(config.outputPath) && ok;
    }

}  // namespace cadence

// Macros compile out under -DCADENCE_DISABLE and cache labels per call site.
#if CADENCE_ENABLED && CADENCE_HAS_CUDA
// CADENCE_KERNEL("label")            -- times work on the default stream
// CADENCE_KERNEL("label", stream)    -- times work on `stream`
#define CADENCE_KERNEL(...) CADENCE_DETAIL_KERNEL_IMPL(__VA_ARGS__, 0, 0)
#define CADENCE_DETAIL_KERNEL_IMPL(label, stream, ...)                                              \
    ::cadence::ScopedKernel CADENCE_DETAIL_UNIQUE(cadenceKernelScope_)(CADENCE_DETAIL_LABEL(label), \
                                                                       stream)

// CADENCE_STAGE records one event per stage. Uninstrumented gaps are charged to
// the following stage; see ScopedStage for the full contract.
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
