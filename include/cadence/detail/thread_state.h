// cadence — per-thread recording state.
//
// Everything a scope touches on the way in and out lives here: the pending
// records, a private cache of CUDA events, and the stage chains. A recording
// thread contends with nobody -- the only other party that ever locks a block
// is Flush(), and then only for as long as it takes to swap two vectors out.
//
// The block outlives its thread. Flush() may be walking it at the moment the
// thread exits, so ownership is shared and the thread merely marks it retired
// on the way out; the registry drops it once it is both retired and drained.
#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

#include "cadence/detail/labels.h"
#include "cadence/detail/platform.h"

namespace cadence {
namespace detail {

// Elapsed times are carried as raw clock ticks and converted to milliseconds at flush. The conversion is only a couple of nanoseconds, but it is a couple of nanoseconds of floating point in the one place this library has no business spending any.
struct HostRecord {
  LabelId label;
  std::int64_t elapsedNs;
};

#if CADENCE_HAS_CUDA
struct DeviceRecord {
  LabelId label;
  // False when `start` is the previous stage's stop event: chained records share their boundary events, and the pool must take each one back exactly once.
  bool ownsStart;
  cudaStream_t stream;
  cudaEvent_t start;
  cudaEvent_t stop;
  // CPU time spent inside the scope, for launch-bound vs compute-bound comparison against the device time.
  std::int64_t hostIssueNs;
};

// The open end of a stage chain on one stream. `tail` is the event that the next stage on this stream will use as its start; `generation` is the flush it belongs to, because a flush releases the events it consumed and any tail from an earlier flush is a dangling handle.
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

  // Guards the pending vectors only. Uncontended in the hot path: the owning thread is the sole writer and Flush() is the sole other reader.
  std::mutex mutex;
  std::vector<HostRecord> pendingHost;
  bool retired = false;

#if CADENCE_HAS_CUDA
  std::vector<DeviceRecord> pendingDevice;
  // Touched only by the owning thread, so deliberately outside `mutex`. Refilled in batches from the registry's shared pool and handed back when the thread retires.
  std::vector<cudaEvent_t> eventCache;
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
