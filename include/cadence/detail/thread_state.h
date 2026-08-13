// cadence: per-thread recording state.
//
// Everything a scope touches on the way in and out lives here
// The block outlives its thread. Flush() may be walking it at the moment the thread exits, so ownership is shared and the thread merely marks it retired on the way out; the registry drops it once it is both retired and drained.
#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

#include "cadence/detail/labels.h"
#include "cadence/detail/platform.h"

namespace cadence {
    namespace detail {

    // Elapsed times are carried as raw clock ticks and converted to milliseconds at flush
    struct HostRecord {
        LabelId label;
        std::int64_t elapsedNs;
    };

#if CADENCE_HAS_CUDA
    struct DeviceRecord {
        LabelId label;
        bool ownsStart; // false when start is prev stage's stop event
        cudaStream_t stream;
        cudaEvent_t start;
        cudaEvent_t stop;
        std::int64_t hostIssueNs; // cpu time spent in scope
    };

    // The open end of a stage chain on one stream. 
    struct StreamChain {
        cudaStream_t stream;
        cudaEvent_t tail;
        std::uint64_t generation;
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
            eventCache.reserve(NUM_EVENTS_PER_REFILL);
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
        std::vector<cudaEvent_t> eventCache; // touched by owning thread
        std::vector<StreamChain> chains;

        StreamChain* FindChain(cudaStream_t stream) {
            for (StreamChain& chain : chains) {
                if (chain.stream == stream) return &chain;
            }
            chains.push_back(StreamChain{stream, nullptr, 0});
            return &chains.back();
        }
#endif
    };
    }  // namespace detail
}  // namespace cadence
