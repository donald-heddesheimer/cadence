// Retained slow iterations and Chrome Trace Event export.
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
        // Position on the shared host/device timeline, or zero when tracing is off.
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

    // Rank by the longer of the enclosing host span and total device work. This
    // handles asynchronous loops whose host scope ends before GPU completion.
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

    // Chrome Trace Event JSON with microsecond timestamps rebased to zero.
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

        // Perfetto reserves tid 0 for the idle task. Shift host and stream lanes
        // by one so each renders on its own track.
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
