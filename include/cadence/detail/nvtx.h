// cadence — NVTX passthrough.
//
// Every scope can also push an NVTX range, so the same instrumentation shows
// up on an Nsight Systems timeline. When no tool is attached the push/pop pair
// is a couple of nanoseconds; when NVTX is absent it compiles to nothing.
#pragma once

#include "cadence/detail/platform.h"

namespace cadence {
namespace detail {

class NvtxRange {
 public:
  explicit NvtxRange(const char* label, bool enabled) : active_(false) {
#if CADENCE_HAS_NVTX
    if (enabled && label) {
      nvtxRangePushA(label);
      active_ = true;
    }
#else
    (void)label;
    (void)enabled;
#endif
  }

  ~NvtxRange() {
#if CADENCE_HAS_NVTX
    if (active_) nvtxRangePop();
#endif
  }

  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;

 private:
  bool active_;
};

}  // namespace detail
}  // namespace cadence
