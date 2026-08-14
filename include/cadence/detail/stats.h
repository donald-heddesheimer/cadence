// cadence: per-label summary statistics.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cadence {
    // Bins in the distribution column of the terminal report. Twelve is a compromise: wide enough that a bimodal label looks bimodal rather than merely wide, narrow enough that the column still fits beside the numbers on an 80-column terminal.
    inline constexpr std::size_t NUM_HISTOGRAM_BINS = 12;

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
        // Sample counts in NUM_HISTOGRAM_BINS equal-width bins spanning min to max. A summary row cannot show that a label is bimodal; this can, and it is the one thing in the report that makes a stall visible rather than merely averaged in. Fixed-width so Stats stays cheap to copy.
        std::array<std::uint32_t, NUM_HISTOGRAM_BINS> histogram{};

        // The deadline this row is held to, and how often it was missed. Zero means no budget applies here, which is the case for every row but the one the budget names.
        double budgetMs = 0.0;
        std::size_t overBudget = 0;

        // True when the run outlived the sample reservoir, so more observations were made than were kept. count, mean, stddev, min, max and overBudget stay exact either way; only the percentiles and the histogram become estimates drawn from a uniform sample of the run.
        bool estimated = false;
    };

    namespace detail {
    // A bounded, uniformly-sampled view of one label's observations, alongside the aggregates that can be maintained exactly in constant space.
    //
    // Everything a deadline is judged on is exact: the count, the extremes, the mean and standard deviation via Welford, and the number of observations over budget. Only the order statistics need the samples themselves, and those come from a reservoir (Vitter's Algorithm R), which holds a uniform sample of the whole run rather than the most recent window. That distinction matters for a profiler: a ring buffer would quietly answer "what did the last few seconds look like" to a question that asked about the run.
    struct SampleSet {
        std::vector<double> reservoir;
        std::uint64_t count = 0;  // Observations kept for statistics; the reservoir holds at most NUM_SAMPLES_RETAINED of them.
        double minMs = 0.0;
        double maxMs = 0.0;
        double mean = 0.0;              // Welford running mean.
        double sumSquaredDelta = 0.0;   // Welford M2; the sum of squared deviations from the running mean.
        std::uint64_t overBudget = 0;   // Observations strictly above the budget in force when they were recorded.
        std::uint64_t rngState = 0;     // Seeded per set so a run is reproducible.

        bool Empty() const { return count == 0; }
        bool Estimated() const { return count > reservoir.size(); }

        // splitmix64. Cheap, seedable, and good enough to place a reservoir index; this is not sampling anything an adversary gets to choose.
        std::uint64_t NextRandom() {
            std::uint64_t z = (rngState += 0x9E3779B97F4A7C15ULL);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            return z ^ (z >> 31);
        }

        // budgetMs of zero means no deadline is configured, so nothing is counted against one.
        void Add(double value, double budgetMs, std::size_t capacity) {
            if (count == 0) {
                minMs = maxMs = value;
            } else {
                if (value < minMs) minMs = value;
                if (value > maxMs) maxMs = value;
            }
            ++count;
            const double delta = value - mean;
            mean += delta / static_cast<double>(count);
            sumSquaredDelta += delta * (value - mean);
            if (budgetMs > 0.0 && value > budgetMs) ++overBudget;

            if (capacity == 0 || reservoir.size() < capacity) {
                reservoir.push_back(value);
                return;
            }
            // Algorithm R: the nth observation replaces a uniformly chosen slot with probability capacity/n, which leaves every observation equally likely to be present.
            const std::uint64_t slot = NextRandom() % count;
            if (slot < capacity) reservoir[static_cast<std::size_t>(slot)] = value;
        }
    };

    // Nearest-rank percentile on an already-sorted range.
    inline double PercentileSorted(const std::vector<double>& sorted, double fraction) {
        if (sorted.empty()) return 0.0;
        const std::size_t rank = static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(sorted.size())));
        const std::size_t index = rank == 0 ? 0 : rank - 1;
        return sorted[std::min(index, sorted.size() - 1)];
    }

    // The exact aggregates come straight off the set; only the percentiles and the histogram are read from the reservoir, which is sorted in place to do it. Named apart from the vector form below rather than overloaded against it, because a braced list of samples would otherwise match both.
    inline Stats ComputeStatsFromSet(const std::string& label, ScopeKind kind, SampleSet samples, std::size_t discarded, double budgetMs = 0.0) {
        Stats stats;
        stats.label = label;
        stats.kind = kind;
        stats.discarded = discarded;
        stats.count = static_cast<std::size_t>(samples.count);
        stats.budgetMs = budgetMs;
        stats.estimated = samples.Estimated();
        if (samples.Empty()) return stats;

        stats.meanMs = samples.mean;
        stats.minMs = samples.minMs;
        stats.maxMs = samples.maxMs;
        stats.jitterMs = stats.maxMs - stats.minMs;
        // Sample standard deviation; undefined for a single observation.
        stats.stddevMs = samples.count > 1 ? std::sqrt(samples.sumSquaredDelta / static_cast<double>(samples.count - 1)) : 0.0;
        if (budgetMs > 0.0) stats.overBudget = static_cast<std::size_t>(samples.overBudget);

        std::vector<double>& kept = samples.reservoir;
        std::sort(kept.begin(), kept.end());
        stats.p50Ms = PercentileSorted(kept, 0.50);
        stats.p95Ms = PercentileSorted(kept, 0.95);

        // Equal-width bins over the observed range. When every sample is identical the range is zero and everything lands in bin 0, which renders as a single spike -- the honest picture of a label that never varied.
        const double span = stats.maxMs - stats.minMs;
        for (double sample : kept) {
            std::size_t bin = 0;
            if (span > 0.0) {
                bin = static_cast<std::size_t>((sample - stats.minMs) / span * static_cast<double>(NUM_HISTOGRAM_BINS));
                if (bin >= NUM_HISTOGRAM_BINS) bin = NUM_HISTOGRAM_BINS - 1;
            }
            ++stats.histogram[bin];
        }
        // The extremes are exact but the reservoir need not have kept them, and the outlier bin is the entire reason this column exists: a lone mark far to the right is how a stall becomes visible rather than averaged in. The minimum and the maximum did occur, so their bins hold at least one observation whether or not the sample retained the evidence.
        if (span > 0.0) {
            if (stats.histogram[0] == 0) stats.histogram[0] = 1;
            if (stats.histogram[NUM_HISTOGRAM_BINS - 1] == 0) stats.histogram[NUM_HISTOGRAM_BINS - 1] = 1;
        }
        return stats;
    }

    // For callers holding a plain vector of observations, which is every test and anyone computing statistics over samples cadence did not gather. Retains all of them, so nothing it returns is an estimate.
    inline Stats ComputeStats(const std::string& label, ScopeKind kind, const std::vector<double>& samples, std::size_t discarded, double budgetMs = 0.0) {
        SampleSet set;
        set.reservoir.reserve(samples.size());
        for (double sample : samples) set.Add(sample, budgetMs, 0);
        return ComputeStatsFromSet(label, kind, std::move(set), discarded, budgetMs);
    }
    }  // namespace detail
}  // namespace cadence
