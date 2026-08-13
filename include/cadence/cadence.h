// cadence — a header-only, drop-in CUDA timing library for real-time loops.
//
// Include this one header. Everything else lives under cadence/detail/.
//
//   #include <cadence/cadence.h>
//
//   for (int step = 0; step < steps; ++step) {
//     { CADENCE_KERNEL("gemm", stream); Gemm<<<grid, block, 0, stream>>>(...); }
//     CADENCE_FLUSH();            // once per loop iteration, not per kernel
//   }
//   CADENCE_REPORT();             // writes cadence.csv
//
// Compile with -DCADENCE_DISABLE and every macro above becomes a no-op with no
// runtime branch left behind.
//
// Scope: single GPU, CUDA runtime API, elapsed time only. For hardware
// counters use `ncu`; for a system-wide timeline use `nsys` (cadence's NVTX
// passthrough feeds it).

#pragma once

#include <cstdlib>
#include <fstream>
#include <ostream>
#include <string>
#include <vector>

#include "cadence/detail/config.h"
#include "cadence/detail/platform.h"
#include "cadence/detail/registry.h"
#include "cadence/detail/report.h"
#include "cadence/detail/scopes.h"
#include "cadence/detail/stats.h"

namespace cadence {

// Apply a configuration. Environment variables still win over whatever is set here; see detail/config.h for the list.
inline void Configure(const Config& config) { detail::Registry::Instance().Configure(config); }

inline const Config& GetConfig() { return detail::Registry::Instance().GetConfig(); }

// Consume pending records: wait on the recorded stop events, compute elapsed times, discard warmup, fold into per-label statistics.
//
// This synchronizes. Call it at a loop or frame boundary -- never between the scopes you are trying to measure.
inline void Flush() { detail::Registry::Instance().Flush(); }

// Per-label statistics for everything flushed so far. Device and host rows are reported separately; a device scope also yields a host row holding its CPU-issue time, so comparing the two shows launch-bound vs compute-bound.
inline std::vector<Stats> Snapshot() { return detail::Registry::Instance().Snapshot(); }

// Drop all accumulated statistics. Warmup counters reset too, so the next N observations per label are discarded again.
inline void Reset() { detail::Registry::Instance().Reset(); }

inline RunInfo QueryRunInfo() { return detail::QueryRunInfo(); }

// Write the report to an already-open stream. Does not flush first.
inline void WriteCsv(std::ostream& out) { detail::Registry::Instance().WriteTo(out); }

// Returns false if the file could not be opened.
inline bool WriteCsv(const std::string& path) {
  std::ofstream out(path);
  if (!out) return false;
  WriteCsv(out);
  return out.good();
}

// Flush, then write to the configured output path. The one call most applications need at shutdown; it also cancels the exit-time fallback write.
inline bool Report() {
  Flush();
  const std::string path = GetConfig().outputPath;
  detail::Registry::Instance().MarkReported();
  if (path.empty()) return true;
  return WriteCsv(path);
}

}  // namespace cadence

// Prefer these over the classes: they are what -DCADENCE_DISABLE removes.

#if CADENCE_ENABLED && CADENCE_HAS_CUDA
// CADENCE_KERNEL("label")            -- times work on the default stream
// CADENCE_KERNEL("label", stream)    -- times work on `stream`
#define CADENCE_KERNEL(...) \
  ::cadence::ScopedKernel CADENCE_DETAIL_UNIQUE(cadenceKernelScope_)(__VA_ARGS__)
#else
#define CADENCE_KERNEL(...) ((void)0)
#endif

#if CADENCE_ENABLED
#define CADENCE_SCOPE(label) ::cadence::ScopedHost CADENCE_DETAIL_UNIQUE(cadenceHostScope_)(label)
#define CADENCE_FLUSH() ::cadence::Flush()
#define CADENCE_REPORT() ::cadence::Report()
#define CADENCE_CONFIGURE(config) ::cadence::Configure(config)
#else
#define CADENCE_SCOPE(label) ((void)0)
#define CADENCE_FLUSH() ((void)0)
#define CADENCE_REPORT() ((void)0)
#define CADENCE_CONFIGURE(config) ((void)0)
#endif
