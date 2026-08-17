// cadence terminal report rendering.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#include "cadence/detail/config.h"
#include "cadence/detail/platform.h"
#include "cadence/detail/stats.h"
#include "cadence/detail/trace.h"

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

    // ANSI escapes are applied after width calculations because they occupy no
    // terminal cells. A value type allows terminal and file outputs to differ.
    struct Theme {
        const char* title = "";
        const char* key = "";     // Provenance keys, headings, rules: present but not competing with the numbers.
        const char* good = "";
        const char* bad = "";
        const char* warn = "";
        const char* accent = "";
        const char* reset = "";
    };

    inline bool StreamIsTerminal(const std::ostream& out) {
        // Generic ostream instances do not expose a file descriptor.
#if defined(_WIN32)
        if (&out == &std::cout) return _isatty(1) != 0;
        if (&out == &std::cerr) return _isatty(2) != 0;
#else
        if (&out == &std::cout) return ::isatty(1) != 0;
        if (&out == &std::cerr) return ::isatty(2) != 0;
#endif
        return false;
    }

    inline Theme SelectTheme(const std::ostream& out, ColorMode mode) {
        const bool colored = mode == ColorMode::Always || (mode == ColorMode::Auto && StreamIsTerminal(out));
        if (!colored) return Theme{};
        // Basic ANSI colors retain compatibility with older terminals.
        Theme theme;
        theme.title = "\033[1m";
        theme.key = "\033[2m";
        theme.good = "\033[32m";
        theme.bad = "\033[1;31m";
        theme.warn = "\033[33m";
        theme.accent = "\033[36m";
        theme.reset = "\033[0m";
        return theme;
    }

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

    // Scale durations to a readable unit with three significant figures.
    inline std::string FormatDuration(double milliseconds, bool unicode) {
        const char* unit = "ms";
        double value = milliseconds;
        if (milliseconds < 0.001) {
            value = milliseconds * 1e6;
            unit = "ns";
        } else if (milliseconds < 1.0) {
            value = milliseconds * 1e3;
            unit = unicode ? "\xc2\xb5s" : "us";  // U+00B5 MICRO SIGN
        } else if (milliseconds >= 1000.0) {
            value = milliseconds / 1e3;
            unit = "s";
        }

        char buffer[32];
        const char* format = value >= 100.0 ? "%.0f%s" : (value >= 10.0 ? "%.1f%s" : "%.2f%s");
        std::snprintf(buffer, sizeof(buffer), format, value, unit);
        return std::string(buffer);
    }

    // Render one cell per histogram bin, scaled against the peak bin.
    inline std::string RenderHistogram(const std::array<std::uint32_t, NUM_HISTOGRAM_BINS>& histogram, bool unicode) {
        static const char* const BLOCKS[8] = {"\xe2\x96\x81", "\xe2\x96\x82", "\xe2\x96\x83", "\xe2\x96\x84",
                                              "\xe2\x96\x85", "\xe2\x96\x86", "\xe2\x96\x87", "\xe2\x96\x88"};
        static const char* const ASCII[8] = {".", ".", ":", ":", "|", "|", "#", "#"};

        std::uint32_t peak = 0;
        for (std::uint32_t count : histogram) peak = std::max(peak, count);
        if (peak == 0) return std::string();

        std::string rendered;
        for (std::uint32_t count : histogram) {
            if (count == 0) {
                rendered += ' ';
                continue;
            }
            // Ceiling division, so any non-empty bin gets at least the shortest mark instead of rounding away to a blank.
            std::size_t level = static_cast<std::size_t>((static_cast<double>(count) / static_cast<double>(peak)) * 7.0 + 0.9999);
            if (level > 7) level = 7;
            rendered += unicode ? BLOCKS[level] : ASCII[level];
        }
        return rendered;
    }

    // Count UTF-8 code points because every glyph emitted here is single-width.
    inline std::size_t DisplayWidth(const std::string& text) {
        std::size_t cells = 0;
        for (unsigned char byte : text) {
            if ((byte & 0xC0) != 0x80) ++cells;
        }
        return cells;
    }

    inline void PadTo(std::string& text, std::size_t width) {
        const std::size_t cells = DisplayWidth(text);
        if (cells < width) text.append(width - cells, ' ');
    }

    inline std::string RightAligned(const std::string& text, std::size_t width) {
        const std::size_t cells = DisplayWidth(text);
        if (cells >= width) return text;
        return std::string(width - cells, ' ') + text;
    }

    // Multi-GPU reports require device-specific columns and summaries.
    inline bool HasMultipleDevices(const std::vector<Stats>& stats) {
        int seen = -1;
        for (const Stats& row : stats) {
            if (row.device < 0) continue;
            if (seen < 0) {
                seen = row.device;
            } else if (row.device != seen) {
                return true;
            }
        }
        return false;
    }

    inline const char* Plural(std::size_t count, const char* singular, const char* plural) {
        return count == 1 ? singular : plural;
    }

    // Record the settings needed to interpret or reproduce a report.
    inline void WriteReportHeader(std::ostream& out, const Config& config, const RunInfo& info, std::size_t failedRecords, std::size_t stalledClockRecords, std::size_t capturedScopes, bool anyEstimated, const Theme& theme) {
        out << theme.title << "cadence report" << theme.reset << "\n";
        out << "  " << theme.key << "device" << theme.reset << "    " << info.deviceName;
        if (info.cudaAvailable) {
            out << " (sm_" << info.computeCapabilityMajor << info.computeCapabilityMinor << ")";
            out << ", sm clock <= " << info.maxSmClockMhz << " MHz, mem clock <= " << info.maxMemClockMhz << " MHz";
        }
        out << "\n";
        if (info.cudaAvailable) {
            out << "  " << theme.key << "clocks" << theme.reset << "    not locked by cadence; run `nvidia-smi -lgc " << info.maxSmClockMhz << "` before comparing runs\n";
        }
        out << "  " << theme.key << "warmup" << theme.reset << "    " << config.warmupIterations << " "
            << Plural(config.warmupIterations, "iteration", "iterations") << " discarded per label\n";
        if (config.sampleEvery > 1) {
            out << "  " << theme.key << "sampling" << theme.reset << "  1 in " << config.sampleEvery << " measured; outliers between samples are not represented\n";
        }
        if (failedRecords > 0) {
            out << "  " << theme.warn << "WARNING" << theme.reset << "   " << failedRecords << " "
                << Plural(failedRecords, "record", "records") << " dropped -- event creation or elapsed-time query failed\n";
        }
        if (stalledClockRecords > 0) {
            out << "  " << theme.warn << "WARNING" << theme.reset << "   " << stalledClockRecords << " host "
                << Plural(stalledClockRecords, "span", "spans") << " dropped -- the monotonic clock did not advance; GPU measurements are unaffected\n";
        }
        if (capturedScopes > 0) {
            out << "  " << theme.warn << "WARNING" << theme.reset << "   " << capturedScopes << " "
                << Plural(capturedScopes, "scope", "scopes") << " skipped during CUDA graph capture; instrument the graph launch instead\n";
        }
        if (anyEstimated) {
            // Identify the fields affected by reservoir sampling.
            out << "  " << theme.key << "sampled" << theme.reset << "   the run outgrew its sample reservoir; n, mean, stddev, min, max and the deadline verdict are exact, p50/p95 and the distribution are estimates\n";
        }
    }

    // Size table columns from their contents.
    inline void WriteStatsTable(std::ostream& out, const std::vector<Stats>& stats, bool unicode, const Theme& theme) {
        if (stats.empty()) {
            out << "\n  no measurements recorded\n";
            return;
        }

        // Show device ids only when the report contains several GPUs.
        const bool multipleDevices = HasMultipleDevices(stats);

        // Left-align text columns and right-align numeric columns.
        constexpr std::size_t NUM_TEXT_COLUMNS = 2;
        std::vector<std::string> headings = {"label", "scope"};
        if (multipleDevices) headings.push_back("dev");
        for (const char* heading : {"n", "mean", "p50", "p95", "max", "jitter"}) headings.push_back(heading);

        std::vector<std::vector<std::string>> rows;
        std::vector<std::string> distributions;
        rows.reserve(stats.size());
        distributions.reserve(stats.size());
        for (const Stats& row : stats) {
            std::vector<std::string> cells;
            cells.reserve(headings.size());
            cells.push_back(row.label);
            cells.push_back(ScopeKindName(row.kind));
            // A plain host span belongs to no GPU, so its cell stays empty rather than claiming device 0.
            if (multipleDevices) cells.push_back(row.device < 0 ? std::string() : std::to_string(row.device));
            cells.push_back(std::to_string(row.count));
            cells.push_back(FormatDuration(row.meanMs, unicode));
            cells.push_back(FormatDuration(row.p50Ms, unicode));
            cells.push_back(FormatDuration(row.p95Ms, unicode));
            cells.push_back(FormatDuration(row.maxMs, unicode));
            cells.push_back(FormatDuration(row.jitterMs, unicode));
            rows.push_back(std::move(cells));
            distributions.push_back(RenderHistogram(row.histogram, unicode));
        }

        std::vector<std::size_t> widths(headings.size());
        for (std::size_t column = 0; column < widths.size(); ++column) widths[column] = DisplayWidth(headings[column]);
        for (const std::vector<std::string>& row : rows) {
            for (std::size_t column = 0; column < widths.size(); ++column) widths[column] = std::max(widths[column], DisplayWidth(row[column]));
        }

        std::string heading = "  ";
        for (std::size_t column = 0; column < widths.size(); ++column) {
            std::string cell = column < NUM_TEXT_COLUMNS ? headings[column] : RightAligned(headings[column], widths[column]);
            PadTo(cell, widths[column]);
            heading += cell + "  ";
        }
        heading += "distribution";

        // The heading is ASCII, so character and byte counts are equal.
        std::string rule;
        for (std::size_t i = 2; i < DisplayWidth(heading); ++i) {
            rule += unicode ? "\xe2\x94\x80" : "-";  // U+2500 BOX DRAWINGS LIGHT HORIZONTAL
        }
        out << "\n" << theme.key << heading << theme.reset << "\n  " << theme.key << rule << theme.reset << "\n";

        for (std::size_t index = 0; index < rows.size(); ++index) {
            out << "  ";
            for (std::size_t column = 0; column < widths.size(); ++column) {
                std::string cell = column < NUM_TEXT_COLUMNS ? rows[index][column] : RightAligned(rows[index][column], widths[column]);
                PadTo(cell, widths[column]);
                out << cell << "  ";
            }
            // Last on the line, so colouring it cannot disturb a column to its right.
            out << theme.accent << distributions[index] << theme.reset << "\n";
        }
    }

    // Summarize device time per iteration. A sole host-only label provides the
    // iteration count; otherwise report only the sum of label means. Weight each
    // device mean by observations per iteration to support repeated labels.
    inline void WriteSummary(std::ostream& out, const std::vector<Stats>& stats, bool unicode, const Theme& theme) {
        // Resolve the iteration denominator before computing device rates.
        const Stats* span = nullptr;
        std::size_t hostOnlyLabels = 0;
        for (const Stats& row : stats) {
            if (row.kind != ScopeKind::Host) continue;
            const bool hasDeviceRow = std::any_of(stats.begin(), stats.end(), [&](const Stats& other) {
                return other.kind == ScopeKind::Device && other.label == row.label;
            });
            if (hasDeviceRow) continue;
            ++hostOnlyLabels;
            span = &row;
        }
        const bool haveIterations = hostOnlyLabels == 1 && span != nullptr && span->count > 0;
        const double iterations = haveIterations ? static_cast<double>(span->count) : 0.0;

        // Concurrent GPUs receive independent totals.
        struct DeviceTotal {
            int device = -1;
            double totalMs = 0.0;
            std::size_t labels = 0;
        };
        std::vector<DeviceTotal> totals;
        for (const Stats& row : stats) {
            if (row.kind != ScopeKind::Device) continue;
            std::size_t index = 0;
            while (index < totals.size() && totals[index].device != row.device) ++index;
            if (index == totals.size()) totals.push_back(DeviceTotal{row.device, 0.0, 0});
            const double perIteration = haveIterations ? static_cast<double>(row.count) / iterations : 1.0;
            totals[index].totalMs += row.meanMs * perIteration;
            ++totals[index].labels;
        }
        if (totals.empty()) return;

        // Keys line up with the provenance block above, which uses the same eight-wide column.
        std::string key = "device";
        PadTo(key, 8);
        for (const DeviceTotal& total : totals) {
            out << "\n  " << theme.key << key << theme.reset << "  " << FormatDuration(total.totalMs, unicode)
                << (haveIterations ? " per iteration across " : " across ") << total.labels << " "
                << Plural(total.labels, "label", "labels")
                << (haveIterations ? "" : ", one mean each");
            if (totals.size() > 1 && total.device >= 0) out << " on device " << total.device;
            out << "\n";
        }

        // Derive host overhead only for one GPU with a known iteration span.
        const double deviceTotalMs = totals.front().totalMs;
        if (totals.size() == 1 && haveIterations && span->meanMs >= deviceTotalMs) {
            const double overheadMs = span->meanMs - deviceTotalMs;
            char share[32];
            std::snprintf(share, sizeof(share), "%.1f%%", 100.0 * deviceTotalMs / span->meanMs);
            key = span->label;
            PadTo(key, 8);
            out << "  " << theme.key << key << theme.reset << "  " << FormatDuration(span->meanMs, unicode) << ", of which " << share << " is GPU work; "
                << FormatDuration(overheadMs, unicode) << " is launch and synchronization\n";
        }
    }

    // Report exact deadline misses and show p95 relative to the budget.
    inline void WriteBudget(std::ostream& out, const std::vector<Stats>& stats, bool unicode, const Theme& theme) {
        // For a multi-GPU label, report the row with the most deadline misses.
        const Stats* held = nullptr;
        std::size_t numHeld = 0;
        for (const Stats& row : stats) {
            if (row.budgetMs <= 0.0 || row.count == 0) continue;
            ++numHeld;
            const bool worse = held == nullptr || row.overBudget > held->overBudget ||
                               (row.overBudget == held->overBudget && row.maxMs > held->maxMs);
            if (worse) held = &row;
        }
        if (held == nullptr) return;

        const std::size_t met = held->count - held->overBudget;
        // Truncate so 100.0% is reserved for runs with no misses.
        const double metPercent = 100.0 * static_cast<double>(met) / static_cast<double>(held->count);
        char metShare[32];
        std::snprintf(metShare, sizeof(metShare), "%.1f%%", std::floor(metPercent * 10.0) / 10.0);
        char worstShare[32];
        std::snprintf(worstShare, sizeof(worstShare), "%.0f%%", 100.0 * held->maxMs / held->budgetMs);

        std::string key = "deadline";
        PadTo(key, 8);
        out << "\n  " << theme.key << key << theme.reset << "  " << FormatDuration(held->budgetMs, unicode) << " on " << held->label << " ("
            << ScopeKindName(held->kind);
        // Only when there was a choice to make, so the single-GPU line is unchanged.
        if (numHeld > 1 && held->device >= 0) out << " on device " << held->device << ", the worst of " << numHeld;
        out << ")\n";

        const bool missed = held->overBudget > 0;
        std::string verdict = missed ? "MISSED" : "met";
        PadTo(verdict, 8);
        // Apply color after padding so escapes do not affect alignment.
        out << "  " << (missed ? theme.bad : theme.good) << verdict << theme.reset << "  " << met << "/" << held->count
            << " iterations inside budget (" << metShare << "); worst " << FormatDuration(held->maxMs, unicode) << " at "
            << worstShare << " of budget\n";

        // Clamp the p95 bar at the deadline width.
        constexpr std::size_t NUM_BAR_CELLS = 40;
        const double share = held->p95Ms / held->budgetMs;
        std::size_t filled = static_cast<std::size_t>(share * static_cast<double>(NUM_BAR_CELLS));
        if (filled > NUM_BAR_CELLS) filled = NUM_BAR_CELLS;

        // Split the bar so each half can use a different color.
        std::string filledBar, emptyBar;
        for (std::size_t cell = 0; cell < NUM_BAR_CELLS; ++cell) {
            if (cell < filled) {
                filledBar += unicode ? "\xe2\x96\x88" : "#";  // U+2588 FULL BLOCK
            } else {
                emptyBar += unicode ? "\xe2\x96\x91" : ".";  // U+2591 LIGHT SHADE
            }
        }
        char p95Share[32];
        std::snprintf(p95Share, sizeof(p95Share), "%.0f%%", 100.0 * share);
        // Color the bar by p95 while the verdict reflects every iteration.
        out << "            [" << (share > 1.0 ? theme.bad : theme.good) << filledBar << theme.reset << theme.key << emptyBar
            << theme.reset << "]  p95 " << FormatDuration(held->p95Ms, unicode) << " at " << p95Share << "\n";
    }

    // Break retained slow iterations down by device stage.
    inline void WriteWorstIterations(std::ostream& out, const std::vector<TraceIteration>& worst, bool unicode, const Theme& theme) {
        if (worst.empty()) return;
        out << "\n  " << theme.key << "slowest iterations" << theme.reset << "\n";
        for (const TraceIteration& iteration : worst) {
            std::string head = "    #" + std::to_string(iteration.index);
            PadTo(head, 10);
            out << theme.accent << head << theme.reset << RightAligned(FormatDuration(iteration.spanMs, unicode), 8) << "  ";
            // Fold device spans by label before limiting the visible stages.
            struct FoldedStage {
                std::string label;
                double totalMs = 0.0;
                std::size_t occurrences = 0;
            };
            std::vector<FoldedStage> folded;
            for (const TraceSpan& span : iteration.spans) {
                if (span.kind != ScopeKind::Device) continue;
                auto found = std::find_if(folded.begin(), folded.end(), [&](const FoldedStage& stage) { return stage.label == span.label; });
                if (found == folded.end()) {
                    folded.push_back(FoldedStage{span.label, span.durationMs, 1});
                } else {
                    found->totalMs += span.durationMs;
                    ++found->occurrences;
                }
            }
            // Sort widest first and preserve execution order for ties.
            std::stable_sort(folded.begin(), folded.end(), [](const FoldedStage& left, const FoldedStage& right) { return left.totalMs > right.totalMs; });

            constexpr std::size_t NUM_STAGES_SHOWN = 8;
            const std::size_t shown = folded.size() < NUM_STAGES_SHOWN ? folded.size() : NUM_STAGES_SHOWN;
            for (std::size_t entry = 0; entry < shown; ++entry) {
                if (entry > 0) out << (unicode ? " \xc2\xb7 " : " | ");  // U+00B7 MIDDLE DOT
                out << folded[entry].label << " " << FormatDuration(folded[entry].totalMs, unicode);
                if (folded[entry].occurrences > 1) out << " x" << folded[entry].occurrences;
            }
            if (folded.empty()) out << "no GPU work recorded";
            if (folded.size() > shown) out << (unicode ? " \xc2\xb7 " : " | ") << "+" << (folded.size() - shown) << " more";
            out << "\n";
        }
    }

    inline void WriteReport(std::ostream& out, const Config& config, const RunInfo& info, const std::vector<Stats>& stats, std::size_t failedRecords, std::size_t stalledClockRecords, std::size_t capturedScopes = 0, const std::vector<TraceIteration>& worst = {}) {
        const bool unicode = config.unicodeOutput;
        // Select color separately for terminal and file streams.
        const Theme theme = SelectTheme(out, config.colorOutput);
        bool anyEstimated = false;
        for (const Stats& row : stats) anyEstimated = anyEstimated || row.estimated;
        WriteReportHeader(out, config, info, failedRecords, stalledClockRecords, capturedScopes, anyEstimated, theme);
        WriteStatsTable(out, stats, unicode, theme);
        WriteBudget(out, stats, unicode, theme);
        WriteWorstIterations(out, worst, unicode, theme);
        WriteSummary(out, stats, unicode, theme);
        out << "\n  " << theme.key << "elapsed time only; for occupancy or bandwidth use `ncu`." << theme.reset << "\n";
        out.flush();
    }

    }  // namespace detail
}  // namespace cadence
