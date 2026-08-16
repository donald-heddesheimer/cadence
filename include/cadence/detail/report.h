// cadence: the terminal report.
//
// The report is the product. A CSV needs a second program before it tells you anything, and a library whose output you have to import somewhere else is a library people instrument once and never read. This renders a block of text meant to be looked at directly: aligned columns, units scaled to the magnitude of the number, and one distribution column per label so a stall is visible rather than averaged into a mean.
//
// Numbers still come with caveats, so the block opens with the device it ran on, the clock state it ran under, and the warmup and sampling settings that produced it.
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

    // ANSI escapes, or empty strings when this report is not going anywhere that understands them. Carried as a value passed down the writers rather than read from a global, because a run that prints to the console and also writes outputPath renders the same report twice and only one of the two wants escape codes in it.
    //
    // Nothing here may reach DisplayWidth() or PadTo(): an escape sequence is ASCII bytes that occupy no cells, so a coloured string measures wider than it draws and every column after it shifts. Colour is applied to whole cells after they are padded, or to the last thing on a line.
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
        // Only the two standard streams can be checked. An ostream gives no way back to a file descriptor, so anything else -- a file, a stringstream -- is treated as not a terminal, which is the safe answer for both.
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
        // Basic ANSI only. The 256-colour and truecolour forms buy a nicer palette in exchange for terminals that render them as garbage, which is a bad trade for output whose whole purpose is being readable.
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

    // Milliseconds are the wrong unit for most of what this library measures: a kernel that takes 0.005025 ms is not readable as five microseconds without counting zeros. Scale to the magnitude instead, and spend the digits where they carry information -- three significant figures is what a timing number is worth given that clocks are not locked.
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

    // One character per histogram bin, scaled against the fullest bin. A bin nobody landed in stays blank rather than showing the shortest block: the gap is the point, because that is what separates a bimodal label from a merely wide one.
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

    // Columns are laid out in terminal cells, and std::string::size() counts bytes. The micro sign in "68.6us" is two bytes and one cell, so byte-based padding shifts every row containing a microsecond figure one column left of its heading. Counting non-continuation bytes gives the code point count, which is the cell count for everything this report emits: digits, ASCII, the micro sign, and the block glyphs are all single-width.
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

    // Whether this report has more than one GPU in it, which is what decides several presentation questions at once: the device column appears, the deadline names the card it is about, and the summary stops adding two cards' spans into a total no iteration ever took.
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

    // Provenance. Anyone quoting a number from this report needs to know what produced it, and a throttled run looks exactly like a healthy one once the numbers are pasted somewhere else.
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
        out << "  " << theme.key << "warmup" << theme.reset << "    " << config.warmupIterations << " iteration(s) discarded per label\n";
        if (config.sampleEvery > 1) {
            // Without this line the counts are unreadable: a 10000-iteration run reporting 1000 samples looks like a bug rather than a setting.
            out << "  " << theme.key << "sampling" << theme.reset << "  1 in " << config.sampleEvery << " measured; outliers between samples are not represented\n";
        }
        if (failedRecords > 0) {
            out << "  " << theme.warn << "WARNING" << theme.reset << "   " << failedRecords << " record(s) dropped -- event creation or elapsed-time query failed\n";
        }
        if (stalledClockRecords > 0) {
            // Worth a warning rather than a silent drop: it says the host timings in this report are thinner than the counts suggest, and it says something true about the machine that the user probably did not know.
            out << "  " << theme.warn << "WARNING" << theme.reset << "   " << stalledClockRecords << " host span(s) dropped -- the monotonic clock did not advance across them, which happens on virtualized hosts; GPU figures are unaffected\n";
        }
        if (capturedScopes > 0) {
            // Not a failure so much as a boundary: the scopes are inside a region cadence cannot instrument, and saying so is what stops the missing rows from reading as an absence of work.
            out << "  " << theme.warn << "WARNING" << theme.reset << "   " << capturedScopes << " scope(s) skipped -- their stream was capturing into a CUDA graph, which cannot carry timing events; wrap the graph launch instead\n";
        }
        if (anyEstimated) {
            // The counts stay honest, so a reader comparing n against the number of iterations needs to be told which columns thinned out and which did not.
            out << "  " << theme.key << "sampled" << theme.reset << "   the run outgrew its sample reservoir; n, mean, stddev, min, max and the deadline verdict are exact, p50/p95 and the distribution are estimates\n";
        }
    }

    // The table. Column widths come from the content rather than from constants, so a long label widens its column instead of wrapping the row.
    inline void WriteStatsTable(std::ostream& out, const std::vector<Stats>& stats, bool unicode, const Theme& theme) {
        if (stats.empty()) {
            out << "\n  no measurements recorded\n";
            return;
        }

        // The device column earns a place only when there is more than one GPU to tell apart. Nearly every run has one, and a column of identical zeroes is a column of noise; a run that has two needs it, because two rows with the same label and different numbers otherwise read as a bug in the report.
        const bool multipleDevices = HasMultipleDevices(stats);

        // Label and scope read as text and belong on the left; every column after them is a number and lines up on the right.
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

        // The rule spans the heading, so its length is a character count rather than a byte count. Everything in the heading is ASCII, which makes the two the same here.
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

    // The one line of arithmetic every reader of this report does by hand, and the only place cadence states a conclusion instead of a measurement.
    //
    // A label that recorded host time but no device time is a plain CADENCE_SCOPE, and when exactly one of those exists it is the loop body: it ran once per pass by construction, so its observation count is the number of iterations measured. That is the denominator the rest of this needs.
    //
    // GPU time per iteration is then each device label's mean weighted by how often that label ran per iteration, which is its own count over that denominator. Summing the unweighted means is only correct when every label fires exactly once per pass, and a graph executor breaks that badly: instrumenting llama.cpp's decode filed 178 matrix multiplies per token under one label, which contributed one mean, and this line reported a workload that is 99.8% GPU-bound as 7.3% GPU-bound. Not imprecise -- backwards, in the one sentence of the report a reader is most likely to quote. docs/case-study.md has the run.
    //
    // Without a loop scope there is no denominator, so the sum of the means is printed as exactly that and the conclusion drawn from it is withheld rather than guessed at.
    inline void WriteSummary(std::ostream& out, const std::vector<Stats>& stats, bool unicode, const Theme& theme) {
        std::size_t deviceLabels = 0;
        for (const Stats& row : stats) {
            if (row.kind == ScopeKind::Device) ++deviceLabels;
        }
        if (totals.empty()) return;

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

        double deviceTotalMs = 0.0;
        for (const Stats& row : stats) {
            if (row.kind != ScopeKind::Device) continue;
            const double perIteration = haveIterations ? static_cast<double>(row.count) / iterations : 1.0;
            deviceTotalMs += row.meanMs * perIteration;
        }

        // Keys line up with the provenance block above, which uses the same eight-wide column.
        std::string key = "device";
        PadTo(key, 8);
        out << "\n  " << theme.key << key << theme.reset << "  " << FormatDuration(deviceTotalMs, unicode)
            << (haveIterations ? " per iteration across " : " across ") << deviceLabels << " label(s)"
            << (haveIterations ? "" : ", one mean each") << "\n";

        if (haveIterations && span->meanMs >= deviceTotalMs) {
            const double overheadMs = span->meanMs - deviceTotalMs;
            char share[32];
            std::snprintf(share, sizeof(share), "%.1f%%", 100.0 * deviceTotalMs / span->meanMs);
            key = span->label;
            PadTo(key, 8);
            out << "  " << theme.key << key << theme.reset << "  " << FormatDuration(span->meanMs, unicode) << ", of which " << share << " is GPU work; "
                << FormatDuration(overheadMs, unicode) << " is launch and synchronization\n";
        }
    }

    // The deadline line. A mean cannot answer "did this loop hold its deadline", because a loop that misses one iteration in fifty has a perfectly healthy mean and a real problem. This answers it as a count, and draws the bar against p95 rather than the mean so the figure shown is one you could plan against.
    inline void WriteBudget(std::ostream& out, const std::vector<Stats>& stats, bool unicode, const Theme& theme) {
        // One label recorded on several GPUs puts the deadline on every one of its rows, and there is only one line to spend. It goes to the worst offender: taking the last row would make the verdict depend on which card happened to sort later, and a deadline line that says "met" while another card missed is the single most misleading thing this report could print.
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
        // Truncated rather than rounded, so a share only reads 100.0% when every iteration actually held. On a long run one miss in 3157 is 99.968%, which %.1f rounds up -- and a line saying MISSED beside 100.0% is precisely the confusion this whole section exists to prevent.
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
        // Coloured after padding: the escapes would otherwise count as cells and push this line out of step with the keys above it.
        out << "  " << (missed ? theme.bad : theme.good) << verdict << theme.reset << "  " << met << "/" << held->count
            << " iterations inside budget (" << metShare << "); worst " << FormatDuration(held->maxMs, unicode) << " at "
            << worstShare << " of budget\n";

        // Filled against p95 and clamped at full: past the deadline the only thing worth reading is that it was passed, and a bar that runs off the line says that more plainly than a longer bar would.
        constexpr std::size_t NUM_BAR_CELLS = 40;
        const double share = held->p95Ms / held->budgetMs;
        std::size_t filled = static_cast<std::size_t>(share * static_cast<double>(NUM_BAR_CELLS));
        if (filled > NUM_BAR_CELLS) filled = NUM_BAR_CELLS;

        // Split so the two halves can take different colours; nothing measures the bar's width afterwards, so escapes are safe inside it.
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
        // The bar is coloured on p95 against the deadline, which is a different question from the verdict above it: a loop can miss occasionally and still have a p95 comfortably inside budget, and that combination is worth seeing at a glance.
        out << "            [" << (share > 1.0 ? theme.bad : theme.good) << filledBar << theme.reset << theme.key << emptyBar
            << theme.reset << "]  p95 " << FormatDuration(held->p95Ms, unicode) << " at " << p95Share << "\n";
    }

    // The slowest passes of the loop, broken down by stage. The table above says how often the loop missed; this says which passes did and where the time went in them, which is the question the table always raises next. A miss is usually one stage rather than everything being slightly slow, and that is visible here and nowhere else in the report.
    inline void WriteWorstIterations(std::ostream& out, const std::vector<TraceIteration>& worst, bool unicode, const Theme& theme) {
        if (worst.empty()) return;
        out << "\n  " << theme.key << "slowest iterations" << theme.reset << "\n";
        for (const TraceIteration& iteration : worst) {
            std::string head = "    #" + std::to_string(iteration.index);
            PadTo(head, 10);
            out << theme.accent << head << theme.reset << RightAligned(FormatDuration(iteration.spanMs, unicode), 8) << "  ";
            // Device spans only. The host-issue side of each label is carried for the trace, where seeing a launch sit ahead of its kernel is the point, but repeating it here would say what the summary table already said.
            //
            // Folded by label and then cut to the widest few, because this line is read, not parsed. One span per stage is the shape this was written for, but a graph executor hands over hundreds: instrumenting llama.cpp per node put roughly 250 spans on one line, several thousand characters of it, three times over, and nothing could be read out of it. Folding first is what makes the cut cheap -- 178 matrix multiplies become one entry carrying their total, which is the number the reader wanted anyway, rather than 178 entries of which eight survive.
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
            // Widest first: the question this section answers is where the time went, and the answer belongs at the front of the line rather than somewhere along it. Stable, so stages that ran for the same length stay in the order they ran.
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
        // Resolved per stream rather than once per run, so the console copy and the outputPath copy of the same report each get what they can render.
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
