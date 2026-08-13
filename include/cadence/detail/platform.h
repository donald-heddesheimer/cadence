// cadence: platform detection and compile-out switches.
//
// This header decides three things, in order of importance:
//   1. Is cadence enabled at all?          (CADENCE_DISABLE)
//   2. Is a CUDA runtime available?        (CADENCE_HAS_CUDA)
//   3. Is NVTX available for passthrough?  (CADENCE_HAS_NVTX)
#pragma once

// Master switch. -DCADENCE_DISABLE reduces every macro in the macro front-end to nothing, mirroring how NVTX compiles out with -DNVTX_DISABLE.
#if defined(CADENCE_DISABLE)
#define CADENCE_ENABLED 0
#else
#define CADENCE_ENABLED 1
#endif

// CUDA runtime
#if !defined(CADENCE_HAS_CUDA)
#if defined(__CUDACC__) || defined(CUDART_VERSION)
#define CADENCE_HAS_CUDA 1
#elif defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#define CADENCE_HAS_CUDA 1
#else
#define CADENCE_HAS_CUDA 0
#endif
#else
#define CADENCE_HAS_CUDA 0
#endif
#endif

#if CADENCE_HAS_CUDA
#include <cuda_runtime.h>
#endif

// NVTX passthrough. NVTX v3 ships with the CUDA Toolkit and is header-only: no library to link.
#if !defined(CADENCE_HAS_NVTX)
#if defined(NVTX_DISABLE)
#define CADENCE_HAS_NVTX 0
#elif defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#define CADENCE_HAS_NVTX 1
#else
#define CADENCE_HAS_NVTX 0
#endif
#else
#define CADENCE_HAS_NVTX 0
#endif
#endif

#if CADENCE_HAS_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

// Token plumbing for the macro front-end: gives each scope macro a unique variable name.
#define CADENCE_DETAIL_CONCAT_IMPL(a, b) a##b
#define CADENCE_DETAIL_CONCAT(a, b) CADENCE_DETAIL_CONCAT_IMPL(a, b)
#define CADENCE_DETAIL_UNIQUE(prefix) CADENCE_DETAIL_CONCAT(prefix, __LINE__)

// Hot-path hints.
#if defined(__GNUC__) || defined(__clang__)
#define CADENCE_ALWAYS_INLINE inline __attribute__((always_inline))
#define CADENCE_LIKELY(expression) __builtin_expect(!!(expression), 1)
#define CADENCE_UNLIKELY(expression) __builtin_expect(!!(expression), 0)
#elif defined(_MSC_VER)
#define CADENCE_ALWAYS_INLINE __forceinline
#define CADENCE_LIKELY(expression) (expression)
#define CADENCE_UNLIKELY(expression) (expression)
#else
#define CADENCE_ALWAYS_INLINE inline
#define CADENCE_LIKELY(expression) (expression)
#define CADENCE_UNLIKELY(expression) (expression)
#endif
