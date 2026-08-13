// cadence: per-label summary statistics.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace cadence {
    enum class ScopeKind {
        Device,  // GPU execution, measured with CUDA events on the work's stream.
        Host,    // CPU span, measured with std::chrono::steady_clock.
    };

    inline const char* ScopeKindName(ScopeKind kind) {
        return kind == ScopeKind::Device ? "device" : "host";
    }

    struct Stats {
        std::string label;
        ScopeKind kind = ScopeKind::Device;
        std::size_t count = 0;      // Samples kept, i.e. after warmup discard.
        std::size_t discarded = 0;  // Samples dropped as warmup.
        double meanMs = 0.0;
        double minMs = 0.0;
        double p50Ms = 0.0;
        double p95Ms = 0.0;
        double maxMs = 0.0;
        double stddevMs = 0.0;
        // Jitter: max - min over the kept samples. For a control loop this is the number that decides whether a deadline holds, not the mean.
        double jitterMs = 0.0;
    };

    namespace detail {
    // Nearest-rank percentile on an already-sorted range.
    inline double PercentileSorted(const std::vector<double>& sorted, double fraction) {
        if (sorted.empty()) return 0.0;
        const std::size_t rank = static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(sorted.size())));
        const std::size_t index = rank == 0 ? 0 : rank - 1;
        return sorted[std::min(index, sorted.size() - 1)];
    }

    // Takes the sample vector by value: it is sorted in place to find percentiles.
    inline Stats ComputeStats(const std::string& label, ScopeKind kind, std::vector<double> samples, std::size_t discarded) {
        Stats stats;
        stats.label = label;
        stats.kind = kind;
        stats.discarded = discarded;
        stats.count = samples.size();
        if (samples.empty()) return stats;

        std::sort(samples.begin(), samples.end());

        double sum = 0.0;
        for (double sample : samples) sum += sample;
        stats.meanMs = sum / static_cast<double>(samples.size());

        double sumSquaredDeviation = 0.0;
        for (double sample : samples) {
            const double deviation = sample - stats.meanMs;
            sumSquaredDeviation += deviation * deviation;
        }
        // Sample standard deviation; undefined for a single observation.
        stats.stddevMs = samples.size() > 1 ? std::sqrt(sumSquaredDeviation / static_cast<double>(samples.size() - 1)) : 0.0;

        stats.minMs = samples.front();
        stats.maxMs = samples.back();
        stats.p50Ms = PercentileSorted(samples, 0.50);
        stats.p95Ms = PercentileSorted(samples, 0.95);
        stats.jitterMs = stats.maxMs - stats.minMs;
        return stats;
    }
    }  // namespace detail
}  // namespace cadence
