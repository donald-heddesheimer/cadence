// cadence — the RAII timers.
//
// ScopedKernel is the heart of the tool: paired CUDA events recorded on the
// same stream as the work, never synchronized in place. ScopedHost is the
// std::chrono equivalent for CPU spans, with the same ergonomics.
#pragma once

#include <chrono>

#include "cadence/detail/nvtx.h"
#include "cadence/detail/platform.h"
#include "cadence/detail/registry.h"

namespace cadence {

// A host-side span measured with steady_clock. Usable in plain C++ translation units with no CUDA in sight.
class ScopedHost {
 public:
  explicit ScopedHost(const char* label)
      : label_(label),
        active_(detail::Registry::Instance().IsEnabled()),
        nvtx_(label, active_ && detail::Registry::Instance().GetConfig().nvtxEnabled),
        start_(std::chrono::steady_clock::now()) {}

  ~ScopedHost() {
    if (!active_) return;
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start_;
    detail::Registry::Instance().RecordHost(label_, elapsed.count());
  }

  ScopedHost(const ScopedHost&) = delete;
  ScopedHost& operator=(const ScopedHost&) = delete;

 private:
  const char* label_;
  bool active_;
  detail::NvtxRange nvtx_;
  std::chrono::steady_clock::time_point start_;
};

#if CADENCE_HAS_CUDA

// Times GPU work with paired CUDA events. Construct it immediately before the launches you care about and let it die immediately after; the events bracket everything enqueued on `stream` in between.
//
// The label must outlive the Flush() that consumes this record. String literals -- what CADENCE_KERNEL passes -- always satisfy that.
class ScopedKernel {
 public:
  explicit ScopedKernel(const char* label, cudaStream_t stream = 0)
      : label_(label),
        stream_(stream),
        start_(nullptr),
        stop_(nullptr),
        active_(detail::Registry::Instance().IsEnabled()),
        nvtx_(label, active_ && detail::Registry::Instance().GetConfig().nvtxEnabled) {
    if (!active_) return;
    detail::Registry& registry = detail::Registry::Instance();
    start_ = registry.AcquireEvent();
    stop_ = registry.AcquireEvent();
    if (!start_ || !stop_) {
      // Out of events: stay quiet rather than taking the application down.
      active_ = false;
      registry.RecordDevice(detail::DeviceRecord{label_, stream_, start_, stop_, 0.0});
      start_ = stop_ = nullptr;
      return;
    }
    hostStart_ = std::chrono::steady_clock::now();
    cudaEventRecord(start_, stream_);
  }

  ~ScopedKernel() {
    if (!active_) return;
    cudaEventRecord(stop_, stream_);
    const auto hostEnd = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> issue = hostEnd - hostStart_;
    detail::Registry::Instance().RecordDevice(
        detail::DeviceRecord{label_, stream_, start_, stop_, issue.count()});
  }

  ScopedKernel(const ScopedKernel&) = delete;
  ScopedKernel& operator=(const ScopedKernel&) = delete;

 private:
  const char* label_;
  cudaStream_t stream_;
  cudaEvent_t start_;
  cudaEvent_t stop_;
  bool active_;
  detail::NvtxRange nvtx_;
  std::chrono::steady_clock::time_point hostStart_;
};

#endif  // CADENCE_HAS_CUDA

}  // namespace cadence
