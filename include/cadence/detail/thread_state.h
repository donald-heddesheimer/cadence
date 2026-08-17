// Per-thread recording state. Shared ownership keeps a block alive during a
// concurrent flush; retired blocks are removed after they drain.
#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <vector>

#include "cadence/detail/labels.h"
#include "cadence/detail/platform.h"

namespace cadence {
    namespace detail {

    // Absolute steady-clock timestamps used for shared host/device timelines.
    CADENCE_ALWAYS_INLINE std::int64_t NowNs() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // Convert raw clock durations to milliseconds during flush.
    struct HostRecord {
        LabelId label;
        std::int64_t startNs;  // steady_clock since its epoch. Carried so a span can be placed on a timeline, not just measured.
        std::int64_t elapsedNs;
    };

#if CADENCE_HAS_CUDA
    struct DeviceRecord {
        LabelId label;
        bool ownsStart;  // False when start is the previous stage's stop event.
        int device;      // Device that owns the events.
        cudaStream_t stream;
        cudaEvent_t start;
        cudaEvent_t stop;
        std::int64_t hostStartNs;  // When the CPU began issuing, on the same clock as HostRecord.
        std::int64_t hostIssueNs;  // CPU time spent in the scope.
    };

    // The open end of a stage chain on one stream.
    struct StreamChain {
        cudaStream_t stream;
        cudaEvent_t tail;
        std::uint64_t generation;
        int device;  // Device that owns the tail event.
        // Make one sampling decision for the complete chain.
        std::uint64_t sampleGeneration;
        bool sampled;
    };

    // A thread's private event supply for one device.
    struct EventCache {
        int device;
        std::vector<cudaEvent_t> events;
    };
#endif

    // Reserve hot-path storage to avoid steady-state allocations.
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

        // Guards pending record vectors only.
        std::mutex mutex;
        std::vector<HostRecord> pendingHost;
        bool retired = false;

#if CADENCE_HAS_CUDA
        std::vector<DeviceRecord> pendingDevice;
        std::vector<EventCache> eventCaches;  // Accessed by the owning thread.
        std::vector<StreamChain> chains;

        // Device and stream counts are small enough for linear lookup.
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
