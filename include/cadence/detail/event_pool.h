// cadence — recycled CUDA event storage.
//
// cudaEventCreate is far too expensive to call per scope, so events are
// created once and handed back to a free list when a record is consumed.
// Events are timing-enabled by design: cudaEventDisableTiming is only correct
// for events used purely for dependency ordering.
#pragma once

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
    // Best effort: at static-destruction time the CUDA runtime may already be gone, and there is nothing useful to do about a failure here.
    for (cudaEvent_t event : free_) cudaEventDestroy(event);
  }

  // Returns nullptr if the runtime refuses to create an event; callers must treat that as "this scope is not measured" rather than as fatal.
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

  void Release(cudaEvent_t event) {
    if (event) free_.push_back(event);
  }

  std::size_t LiveCount() const { return liveCount_; }

 private:
  std::vector<cudaEvent_t> free_;
  std::size_t liveCount_ = 0;
};

}  // namespace detail
}  // namespace cadence

#endif  // CADENCE_HAS_CUDA
