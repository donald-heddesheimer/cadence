// cadence example — a control-loop shaped workload.
//
// The point is not the arithmetic; it is the shape. A robotics or perception
// loop runs the same handful of kernels every iteration, and the question is
// never "how fast is this kernel in isolation" but "what does iteration N cost,
// and how much does it vary". That is what cadence reports.
//
// Build:
//   export CUDACXX=/usr/local/cuda-12.9/bin/nvcc
//   cmake -B build && cmake --build build
//   ./build/examples/cadence_example_loop
//
// Then read cadence.csv.

#include <cadence/cadence.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int NUM_ELEMENTS = 1 << 20;
constexpr int NUM_ITERATIONS = 200;
constexpr int NUM_THREADS_PER_BLOCK = 256;

__global__ void Saxpy(float alpha, const float* x, float* y, int numElements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numElements) y[index] = alpha * x[index] + y[index];
}

// Stands in for the second stage of a pipeline, so the report has more than one
// label to compare.
__global__ void Scale(float factor, float* y, int numElements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numElements) y[index] *= factor;
}

bool CudaOk(cudaError_t status, const char* what) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(status));
  return false;
}

}  // namespace

int main() {
  cadence::Config config;
  config.warmupIterations = 10;  // Discard context creation and autotuning.
  config.outputPath = "cadence.csv";
  cadence::Configure(config);

  float* deviceX = nullptr;
  float* deviceY = nullptr;
  const std::size_t bytes = NUM_ELEMENTS * sizeof(float);
  if (!CudaOk(cudaMalloc(&deviceX, bytes), "cudaMalloc(x)")) return 1;
  if (!CudaOk(cudaMalloc(&deviceY, bytes), "cudaMalloc(y)")) return 1;

  cudaStream_t stream = nullptr;
  if (!CudaOk(cudaStreamCreate(&stream), "cudaStreamCreate")) return 1;

  const std::vector<float> hostX(NUM_ELEMENTS, 1.0f);
  if (!CudaOk(cudaMemcpy(deviceX, hostX.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy")) {
    return 1;
  }
  if (!CudaOk(cudaMemset(deviceY, 0, bytes), "cudaMemset")) return 1;

  const int numBlocks = (NUM_ELEMENTS + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

  for (int iteration = 0; iteration < NUM_ITERATIONS; ++iteration) {
    {
      // One host scope wrapping the whole iteration: this is the loop latency
      // a control engineer actually cares about.
      CADENCE_SCOPE("iteration");

      {
        CADENCE_KERNEL("saxpy", stream);
        Saxpy<<<numBlocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(2.0f, deviceX, deviceY,
                                                               NUM_ELEMENTS);
      }
      {
        CADENCE_KERNEL("scale", stream);
        Scale<<<numBlocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(0.5f, deviceY, NUM_ELEMENTS);
      }

      // A real loop synchronizes somewhere -- here, because the next iteration
      // depends on the result. cadence never forced it.
      cudaStreamSynchronize(stream);
    }

    // Flush once per iteration, outside the measured region.
    CADENCE_FLUSH();
  }

  if (!CudaOk(cudaGetLastError(), "kernel launch")) return 1;

  CADENCE_REPORT();

  for (const cadence::Stats& row : cadence::Snapshot()) {
    std::printf("%-12s %-7s n=%-5zu mean=%8.4f ms  p95=%8.4f ms  jitter=%8.4f ms\n",
                row.label.c_str(), cadence::ScopeKindName(row.kind), row.count, row.meanMs,
                row.p95Ms, row.jitterMs);
  }
  std::printf("\nwrote %s\n", config.outputPath.c_str());

  cudaStreamDestroy(stream);
  cudaFree(deviceX);
  cudaFree(deviceY);
  return 0;
}
