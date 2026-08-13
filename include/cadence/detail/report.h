// Numbers come with caveats, so every report carries a header describing the device it ran on, the clock state it ran under, and the warmup setting that produced it.
#pragma once

#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "cadence/detail/config.h"
#include "cadence/detail/platform.h"
#include "cadence/detail/stats.h"

namespace cadence {

    struct RunInfo {
        std::string deviceName = "unknown";
        int computeCapabilityMajor = 0;
        int computeCapabilityMinor = 0;
        int maxSmClockMhz = 0;  // Advertised maximum, not the clock actually used.
        int maxMemClockMhz = 0;
        bool cudaAvailable = false;
    };

    namespace detail {

    inline RunInfo QueryRunInfo() {
        RunInfo info;
#if CADENCE_HAS_CUDA
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) return info;
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) return info;
        info.cudaAvailable = true;
        info.deviceName = properties.name;
        info.computeCapabilityMajor = properties.major;
        info.computeCapabilityMinor = properties.minor;
        info.maxSmClockMhz = properties.clockRate / 1000;
        info.maxMemClockMhz = properties.memoryClockRate / 1000;
#endif
        return info;
    }

    // Comment block above the CSV rows. `#`-prefixed so pandas/awk skip it with their usual comment handling and the file stays a single artifact.
    inline void WriteReportHeader(std::ostream& out, const Config& config, const RunInfo& info, std::size_t failedRecords) {
        out << "# cadence report\n";
        out << "# device: " << info.deviceName;
        if (info.cudaAvailable)  out << " (sm_" << info.computeCapabilityMajor << info.computeCapabilityMinor << ")";
        out << "\n";
        out << "# max_sm_clock_mhz: " << info.maxSmClockMhz << "  max_mem_clock_mhz: " << info.maxMemClockMhz << "\n";
        out << "# clock_state: not locked by cadence; boost clocks drift run to run.\n";
        out << "#   lock with `nvidia-smi -lgc <mhz>` before comparing runs.\n";
        out << "# warmup_iterations_discarded_per_label: " << config.warmupIterations << "\n";
        if (config.sampleEvery > 1) {
            out << "# sample_every: " << config.sampleEvery << " -- one observation in " << config.sampleEvery << " was measured; outliers between samples are not represented.\n";
        }
        if (failedRecords > 0) {
            out << "# warning: " << failedRecords << " record(s) dropped -- event creation or elapsed-time query failed\n";
        }
        out << "# elapsed time only; for occupancy or bandwidth use `ncu`.\n";
    }

    inline void WriteStatsCsv(std::ostream& out, const std::vector<Stats>& stats) {
        out << "label,scope,count,warmup_discarded,mean_ms,min_ms,p50_ms,p95_ms,max_ms," "stddev_ms,jitter_ms\n";
        out << std::fixed << std::setprecision(6);
        for (const Stats& row : stats) {
            out << row.label << ',' << ScopeKindName(row.kind) << ',' << row.count << ',' << row.discarded
                << ',' << row.meanMs << ',' << row.minMs << ',' << row.p50Ms << ',' << row.p95Ms << ','
                << row.maxMs << ',' << row.stddevMs << ',' << row.jitterMs << '\n';
        }
    }

    }  // namespace detail
}  // namespace cadence
