// cadence — the RAII timers.
//
// Three of them, and the difference between them is how many CUDA events they
// record, because that is what the instrumentation costs.
//
//   ScopedHost    no events. A steady_clock bracket around a CPU span.
//   ScopedKernel  two events. Measures exactly the work inside the scope.
//   ScopedStage   one event. Measures from the end of the previous stage.
//
// A timing-enabled cudaEventRecord costs roughly 1.4 us of host time and that
// is a property of the CUDA API, not of this library: the same call with
// cudaEventDisableTiming costs a tenth of it, because the timestamp is the
// expensive part. Two of them dominate everything else a device scope does by
// an order of magnitude. ScopedStage exists because halving the number of
// records is the only way to make a real dent in that, and it is not free --
// see the contract on the class.
//
// None of them synchronize. Every one of them appends to a thread-local buffer
// and returns.
#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>

#include "cadence/detail/labels.h"
#include "cadence/detail/nvtx.h"
#include "cadence/detail/platform.h"
#include "cadence/detail/registry.h"

namespace cadence {
namespace detail {

CADENCE_ALWAYS_INLINE std::int64_t ElapsedNs(std::chrono::steady_clock::time_point from,
                                             std::chrono::steady_clock::time_point to) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count();
}

}  // namespace detail

// A host-side span measured with steady_clock. Usable in plain C++ translation units with no CUDA in sight.
class ScopedHost {
 public:
  explicit ScopedHost(const detail::LabelHandle& label)
      : labelId_(label.id),
        active_(detail::hotConfig.enabled.load(std::memory_order_relaxed) &&
                detail::ShouldSample(label)),
        nvtx_(label.name,
              active_ && detail::hotConfig.nvtxEnabled.load(std::memory_order_relaxed)) {
    if (CADENCE_LIKELY(active_)) start_ = std::chrono::steady_clock::now();
  }

  // Interns on every construction, and takes a lock to do it. CADENCE_SCOPE resolves the label once per call site instead; prefer it.
  explicit ScopedHost(const char* label)
      : ScopedHost(detail::LabelTable::Instance().Intern(label)) {}

  ~ScopedHost() {
    if (!active_) return;
    const auto end = std::chrono::steady_clock::now();
    const std::int64_t elapsedNs = detail::ElapsedNs(start_, end);
    detail::ThreadState& state = detail::TlsState();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.pendingHost.push_back(detail::HostRecord{labelId_, elapsedNs});
  }

  ScopedHost(const ScopedHost&) = delete;
  ScopedHost& operator=(const ScopedHost&) = delete;

 private:
  // Only the id survives construction. Holding a pointer to the handle instead would dangle for the const char* constructor above, whose handle is a temporary that dies when this constructor returns, and the name is needed only long enough for NvtxRange to push it.
  detail::LabelId labelId_;
  bool active_;
  detail::NvtxRange nvtx_;
  std::chrono::steady_clock::time_point start_{};
};

#if CADENCE_HAS_CUDA

// Times GPU work with paired CUDA events. Construct it immediately before the launches you care about and let it die immediately after; the events bracket everything enqueued on `stream` in between.
//
// This is the honest one: it measures the work inside the scope and nothing else, whatever else the stream is doing.
class ScopedKernel {
 public:
  explicit ScopedKernel(const detail::LabelHandle& label, cudaStream_t stream = 0)
      : labelId_(label.id),
        stream_(stream),
        active_(detail::hotConfig.enabled.load(std::memory_order_relaxed) &&
                detail::ShouldSample(label)),
        nvtx_(label.name,
              active_ && detail::hotConfig.nvtxEnabled.load(std::memory_order_relaxed)) {
    if (!active_) return;
    detail::ThreadState& state = detail::TlsState();
    start_ = detail::TakeEvent(state);
    stop_ = detail::TakeEvent(state);
    if (CADENCE_UNLIKELY(!start_ || !stop_)) {
      // Out of events: stay quiet rather than taking the application down, but hand back whatever was taken so the count stays honest.
      active_ = false;
      Append(state, detail::DeviceRecord{labelId_, true, stream_, start_, stop_, 0});
      start_ = stop_ = nullptr;
      return;
    }
    hostStart_ = std::chrono::steady_clock::now();
    cudaEventRecord(start_, stream_);
  }

  explicit ScopedKernel(const char* label, cudaStream_t stream = 0)
      : ScopedKernel(detail::LabelTable::Instance().Intern(label), stream) {}

  ~ScopedKernel() {
    if (!active_) return;
    cudaEventRecord(stop_, stream_);
    const auto hostEnd = std::chrono::steady_clock::now();
    detail::ThreadState& state = detail::TlsState();
    Append(state, detail::DeviceRecord{labelId_, true, stream_, start_, stop_,
                                       detail::ElapsedNs(hostStart_, hostEnd)});
  }

  ScopedKernel(const ScopedKernel&) = delete;
  ScopedKernel& operator=(const ScopedKernel&) = delete;

 private:
  static void Append(detail::ThreadState& state, const detail::DeviceRecord& record) {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.pendingDevice.push_back(record);
  }

  detail::LabelId labelId_;
  cudaStream_t stream_;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
  bool active_;
  detail::NvtxRange nvtx_;
  std::chrono::steady_clock::time_point hostStart_{};
};

// Times one stage of a chain, using the previous stage's stop event as its start. One record per stage plus one to open the chain, instead of two per stage -- measured at roughly a third off the instrumentation cost of a five-stage loop.
//
// The contract, which is the whole of the difference: a stage measures from the end of the previous stage on its stream, not from its own opening brace. Anything enqueued on that stream between two stages is charged to the later one. Use it when the loop is fully instrumented -- every launch on the stream inside some stage -- and use ScopedKernel when it is not, or when one stage's number has to stand on its own.
//
// Chains are per thread and per stream, and every Flush() breaks them, so a stage never measures across a loop boundary. Sampling does not apply here: a skipped stage would silently fold its time into the next one.
class ScopedStage {
 public:
  explicit ScopedStage(const detail::LabelHandle& label, cudaStream_t stream = 0)
      : labelId_(label.id),
        stream_(stream),
        active_(detail::hotConfig.enabled.load(std::memory_order_relaxed)),
        nvtx_(label.name,
              active_ && detail::hotConfig.nvtxEnabled.load(std::memory_order_relaxed)) {
    if (!active_) return;
    detail::ThreadState& state = detail::TlsState();
    detail::StreamChain* chain = state.FindChain(stream_);
    generation_ = detail::Registry::Instance().FlushGeneration();
    // A tail from an earlier flush is a handle to an event already back in the pool, so age is checked before the pointer is trusted.
    if (chain->generation != generation_ || !chain->tail) {
      cudaEvent_t head = detail::TakeEvent(state);
      if (CADENCE_UNLIKELY(!head)) {
        active_ = false;
        chain->tail = nullptr;
        return;
      }
      cudaEventRecord(head, stream_);
      chain->tail = head;
      chain->generation = generation_;
      ownsStart_ = true;
    }
    start_ = chain->tail;
    hostStart_ = std::chrono::steady_clock::now();
  }

  explicit ScopedStage(const char* label, cudaStream_t stream = 0)
      : ScopedStage(detail::LabelTable::Instance().Intern(label), stream) {}

  ~ScopedStage() {
    if (!active_) return;
    detail::ThreadState& state = detail::TlsState();
    cudaEvent_t stop = detail::TakeEvent(state);
    const auto hostEnd = std::chrono::steady_clock::now();
    detail::StreamChain* chain = state.FindChain(stream_);
    if (CADENCE_UNLIKELY(!stop)) {
      // Break the chain rather than leave a head nobody will ever release; the record carries the orphan back to the pool via Flush().
      chain->tail = nullptr;
      if (ownsStart_) {
        Append(state, detail::DeviceRecord{labelId_, true, stream_, start_, nullptr, 0});
      }
      return;
    }
    cudaEventRecord(stop, stream_);
    chain->tail = stop;
    chain->generation = generation_;
    Append(state, detail::DeviceRecord{labelId_, ownsStart_, stream_, start_, stop,
                                       detail::ElapsedNs(hostStart_, hostEnd)});
  }

  ScopedStage(const ScopedStage&) = delete;
  ScopedStage& operator=(const ScopedStage&) = delete;

 private:
  static void Append(detail::ThreadState& state, const detail::DeviceRecord& record) {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.pendingDevice.push_back(record);
  }

  detail::LabelId labelId_;
  cudaStream_t stream_;
  cudaEvent_t start_ = nullptr;
  bool active_;
  // True only for the stage that opened the chain: every other stage borrows its predecessor's stop event, and the pool must take each event back exactly once.
  bool ownsStart_ = false;
  std::uint64_t generation_ = 0;
  detail::NvtxRange nvtx_;
  std::chrono::steady_clock::time_point hostStart_{};
};

#endif  // CADENCE_HAS_CUDA

}  // namespace cadence
