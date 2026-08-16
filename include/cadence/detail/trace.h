// cadence: the worst iterations, and the trace file that shows what happened inside them.
//
// The summary table answers "how often did this loop miss", and then stops being useful. The next question is always "which iterations, and what was slow in them", and a distribution cannot answer it: by the time an outlier reaches the histogram it is a single mark with no detail behind it. So the slowest iterations are kept whole, with every stage that ran inside them, and the report prints their breakdown.
//
// The same retained iterations are what gets written as a trace. Exporting an entire run would be both enormous and useless -- nobody scrolls a million spans looking for the bad one -- whereas exporting the handful of iterations that actually missed is small enough to open instantly and is exactly the part worth looking at.
//
// The format is the Chrome Trace Event format, which https://ui.perfetto.dev opens directly. Writing JSON someone else already built a timeline viewer for is a much better trade than building one.
#pragma once

#include <cstdint>
#include <cstdio>
#include <ostream>
#include <string>
#include <vector>

#include "cadence/detail/labels.h"
#include "cadence/detail/platform.h"
#include "cadence/detail/stats.h"

namespace cadence {

    // One label's span inside one iteration, with its label already resolved.
    struct TraceSpan {
        std::string label;
        ScopeKind kind = ScopeKind::Host;
        double durationMs = 0.0;
        // Absolute position on the steady clock, in nanoseconds. GPU spans are placed by measuring backwards from an event whose completion was observed at a known host time, so device and host spans share one timeline. Zero when the run was not building a trace.
        std::int64_t startNs = 0;
        int lane = 0;  // 0 is the host lane; device spans get one lane per stream.
    };

    // Everything recorded between two flushes, which is one pass of the instrumented loop.
    struct TraceIteration {
        std::uint64_t index = 0;  // Which flush produced it, counted from the first.
        double spanMs = 0.0;      // What the iteration is ranked by: the loop scope, or the GPU work when that ran longer. See IterationSpanMs.
        std::vector<TraceSpan> spans;
    };

    namespace detail {

    // The unresolved form, which is what the flush path appends to. Labels stay as ids until the report asks for them, so nothing on the hot path allocates a string.
    struct IterationSpan {
        LabelId label = 0;
        ScopeKind kind = ScopeKind::Host;
        double durationMs = 0.0;
        std::int64_t startNs = 0;
        int lane = 0;
    };

    struct IterationRecord {
        std::uint64_t index = 0;
        double spanMs = 0.0;
        std::vector<IterationSpan> spans;
    };

    // How long an iteration took, for ranking.
    //
    // A CADENCE_SCOPE wrapped around the loop body is the iteration, so the longest pure host span is normally the iteration's own duration and needs no reconstruction from parts. When nothing recorded a host span, adding up the GPU work ranks a device-only run in roughly the right order, which is all a ranking needs, though concurrent streams make it an overestimate.
    //
    // Taking whichever is larger rather than preferring the host span is what keeps this honest when the host scope does not enclose the work it launched. A loop that queues asynchronous work and synchronizes somewhere else has a host span that returns in microseconds against milliseconds of GPU time -- llama.cpp's decode is 8us of host per 3.9ms graph -- and ranking those iterations by the host span ranks them by host jitter. The run that found this picked an iteration for an 18.9us host span with an entirely ordinary GPU pass in it while the slowest GPU iteration in the run never appeared. Where the host scope does enclose the work, it is the larger of the two by construction and nothing changes.
    inline double IterationSpanMs(const std::vector<IterationSpan>& spans) {
        double longestHost = 0.0;
        double deviceTotal = 0.0;
        for (const IterationSpan& span : spans) {
            if (span.kind == ScopeKind::Host) {
                if (span.durationMs > longestHost) longestHost = span.durationMs;
            } else {
                deviceTotal += span.durationMs;
            }
        }
        return longestHost > deviceTotal ? longestHost : deviceTotal;
    }

    // JSON string escaping, for label text that came from the caller.
    inline std::string EscapeJson(const std::string& text) {
        std::string escaped;
        escaped.reserve(text.size() + 2);
        for (char character : text) {
            switch (character) {
                case '"': escaped += "\\\""; break;
                case '\\': escaped += "\\\\"; break;
                case '\n': escaped += "\\n"; break;
                case '\r': escaped += "\\r"; break;
                case '\t': escaped += "\\t"; break;
                default:
                    if (static_cast<unsigned char>(character) < 0x20) {
                        char buffer[8];
                        std::snprintf(buffer, sizeof(buffer), "\\u%04x", character);
                        escaped += buffer;
                    } else {
                        escaped += character;
                    }
            }
        }
        return escaped;
    }

    // Chrome Trace Event JSON. Timestamps are microseconds, which is what the format specifies, and are rebased so the earliest span in the file sits at zero: the steady clock's own origin is arbitrary and a viewer showing timestamps in the billions helps nobody.
    inline void WriteTraceJson(std::ostream& out, const std::vector<TraceIteration>& iterations) {
        std::int64_t origin = 0;
        bool haveOrigin = false;
        int maxLane = 0;
        for (const TraceIteration& iteration : iterations) {
            for (const TraceSpan& span : iteration.spans) {
                if (!haveOrigin || span.startNs < origin) {
                    origin = span.startNs;
                    haveOrigin = true;
                }
                if (span.lane > maxLane) maxLane = span.lane;
            }
        }

        out << "{\n  \"displayTimeUnit\": \"ns\",\n  \"traceEvents\": [\n";
        bool first = true;
        const auto separator = [&]() {
            if (!first) out << ",\n";
            first = false;
        };

        // Lane numbers are shifted by one on the way out, because thread id 0 is not a thread. Perfetto inherits the kernel's convention where tid 0 is the idle task, and a track claiming that id is folded into another rather than drawn on its own row: the host spans then land in a device stream's track and nest inside it by time containment, which turns the one thing a timeline is for -- a launch sitting visibly ahead of the kernel it queued -- into a stack of boxes that looks like a call tree. Host becomes tid 1, stream N becomes tid N+1.
        constexpr int TID_OFFSET = 1;

        // Metadata naming the lanes. Without these a viewer labels every row by a bare numeric id.
        separator();
        out << "    {\"ph\":\"M\",\"pid\":1,\"name\":\"process_name\",\"args\":{\"name\":\"cadence\"}}";
        separator();
        out << "    {\"ph\":\"M\",\"pid\":1,\"tid\":" << TID_OFFSET << ",\"name\":\"thread_name\",\"args\":{\"name\":\"host (CPU)\"}}";
        for (int lane = 1; lane <= maxLane; ++lane) {
            separator();
            out << "    {\"ph\":\"M\",\"pid\":1,\"tid\":" << lane + TID_OFFSET
                << ",\"name\":\"thread_name\",\"args\":{\"name\":\"device stream " << lane << "\"}}";
        }

        for (const TraceIteration& iteration : iterations) {
            for (const TraceSpan& span : iteration.spans) {
                separator();
                const double startUs = static_cast<double>(span.startNs - origin) / 1000.0;
                const double durationUs = span.durationMs * 1000.0;
                char numbers[128];
                std::snprintf(numbers, sizeof(numbers), "\"ts\":%.3f,\"dur\":%.3f", startUs, durationUs);
                out << "    {\"ph\":\"X\",\"pid\":1,\"tid\":" << span.lane + TID_OFFSET
                    << ",\"name\":\"" << EscapeJson(span.label) << "\""
                    << ",\"cat\":\"" << ScopeKindName(span.kind) << "\","
                    << numbers
                    << ",\"args\":{\"iteration\":" << iteration.index << "}}";
            }
        }
        out << "\n  ]\n}\n";
    }

    }  // namespace detail
}  // namespace cadence
