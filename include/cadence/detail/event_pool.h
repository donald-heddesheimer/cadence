// cadence: recycled CUDA event storage.
//
// cudaEventCreate is far too expensive to call per scope, so events are created once and handed back to a free list when a record is consumed. Events are timing-enabled by design: cudaEventDisableTiming is only correct for events used purely for dependency ordering, and it is not a shortcut available here -- it is a tenth the cost precisely because it skips the timestamp this library exists to read.
//
// This pool is the shared one, behind the registry's lock. Threads draw from it in batches into a thread-local cache, so the lock is taken once per refill rather than twice per scope.
#pragma once

#include <cstddef>
#include <vector>

#include "cadence/detail/platform.h"

#if CADENCE_HAS_CUDA

namespace cadence {
    namespace detail {
    class EventPool {
       public:
        EventPool() = default;
        EventPool(const EventPool&) = delete;
        EventPool& operator=(const EventPool&) = delete;

        ~EventPool() {
            // at static-destruction time the CUDA runtime may already be gone, and there is nothing useful to do about a failure here.
            for (cudaEvent_t event : free_) cudaEventDestroy(event);
        }

        // Returns nullptr if the runtime refuses to create an event
        cudaEvent_t Acquire() {
            if (!free_.empty()) {
                cudaEvent_t event = free_.back();
                free_.pop_back();
                return event;
            }
            cudaEvent_t event = nullptr;
            if (cudaEventCreate(&event) != cudaSuccess) return nullptr;
            ++liveCount_;
            return event;
        }

        // Moves up to count events into out, creating whatever the free list cannot supply.
        std::size_t AcquireInto(std::vector<cudaEvent_t>& out, std::size_t count) {
            std::size_t added = 0;
            while (added < count && !free_.empty()) {
                out.push_back(free_.back());
                free_.pop_back();
                ++added;
            }
            while (added < count) {
                cudaEvent_t event = nullptr;
                if (cudaEventCreate(&event) != cudaSuccess) break;
                ++liveCount_;
                out.push_back(event);
                ++added;
            }
            return added;
        }

        void Release(cudaEvent_t event) {
            if (event) free_.push_back(event);
        }

        void ReleaseAll(std::vector<cudaEvent_t>& events) {
            for (cudaEvent_t event : events) {
                if (event) free_.push_back(event);
            }
            events.clear();
        }

        std::size_t LiveCount() const { return liveCount_; }

       private:
        std::vector<cudaEvent_t> free_;
        std::size_t liveCount_ = 0;
    };

    }  // namespace detail
}  // namespace cadence

#endif  // CADENCE_HAS_CUDA
