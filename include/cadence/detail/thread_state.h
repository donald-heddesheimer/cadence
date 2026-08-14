// cadence: per-thread recording state.
//
// Everything a scope touches on the way in and out lives here
// The block outlives its thread. Flush() may be walking it at the moment the thread exits, so ownership is shared and the thread merely marks it retired on the way out; the registry drops it once it is both retired and drained.
#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <vector>

#include "cadence/detail/labels.h"
#include "cadence/detail/platform.h"

namespace cadence {
    namespace detail {

    // Absolute nanoseconds on the steady clock. Spans need a position as well as a length before they can be drawn on a timeline, and one origin shared by every scope is what lets a host span and a GPU span be compared at all. Lives beside the records that store its result.
    CADENCE_ALWAYS_INLINE std::int64_t NowNs() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // Elapsed times are carried as raw clock ticks and converted to milliseconds at flush
    struct HostRecord {
        LabelId label;
        std::int64_t startNs;  // steady_clock since its epoch. Carried so a span can be placed on a timeline, not just measured.
        std::int64_t elapsedNs;
    };

#if CADENCE_HAS_CUDA
    struct DeviceRecord {
        LabelId label;
        bool ownsStart; // false when start is prev stage's stop event
        int device;     // Device the events were created on; they go back to that device's pool and nowhere else.
        cudaStream_t stream;
        cudaEvent_t start;
        cudaEvent_t stop;
        std::int64_t hostStartNs;  // When the CPU began issuing, on the same clock as HostRecord.
        std::int64_t hostIssueNs;  // cpu time spent in scope
    };

    // The open end of a stage chain on one stream.
    struct StreamChain {
        cudaStream_t stream;
        cudaEvent_t tail;
        std::uint64_t generation;
        int device;                        // Device the tail was created on; a change breaks the chain rather than pairing events across devices.
        // Sampling is decided once per chain rather than once per stage: a chain whose middle links are missing does not measure the stages that remain, it measures them plus the gaps where the skipped ones used to be.
        std::uint64_t sampleGeneration;
        bool sampled;
    };

    // A thread's private event supply for one device.
    struct EventCache {
        int device;
        std::vector<cudaEvent_t> events;
    };
#endif

    // Grown once, then reused. A reallocation in the middle of a control loop shows up as jitter in the very measurement it is taken to support.
    inline constexpr std::size_t NUM_RECORDS_RESERVED = 256;
    inline constexpr std::size_t NUM_EVENTS_PER_REFILL = 64;

    struct ThreadState {
        ThreadState() {
            pendingHost.reserve(NUM_RECORDS_RESERVED);
#if CADENCE_HAS_CUDA
            pendingDevice.reserve(NUM_RECORDS_RESERVED);
#endif
        }

        ThreadState(const ThreadState&) = delete;
        ThreadState& operator=(const ThreadState&) = delete;

        // Guards the pending vectors only
        std::mutex mutex;
        std::vector<HostRecord> pendingHost;
        bool retired = false;

#if CADENCE_HAS_CUDA
        std::vector<DeviceRecord> pendingDevice;
        std::vector<EventCache> eventCaches; // touched by owning thread
        std::vector<StreamChain> chains;

        // Both lists hold one entry per device or stream this thread has actually touched, which in practice is one or two, so a scan beats a hash lookup on the hot path.
        std::vector<cudaEvent_t>& CacheFor(int device) {
            for (EventCache& cache : eventCaches) {
                if (cache.device == device) return cache.events;
            }
            eventCaches.push_back(EventCache{device, {}});
            eventCaches.back().events.reserve(NUM_EVENTS_PER_REFILL);
            return eventCaches.back().events;
        }

        StreamChain* FindChain(cudaStream_t stream) {
            for (StreamChain& chain : chains) {
                if (chain.stream == stream) return &chain;
            }
            chains.push_back(StreamChain{stream, nullptr, 0, -1, 0, true});
            return &chains.back();
        }
#endif
    };
    }  // namespace detail
}  // namespace cadence
