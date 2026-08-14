// cadence example: a control-loop shaped workload.
//
// Build:
//   export CUDACXX=/usr/local/cuda-12.9/bin/nvcc
//   cmake -B build && cmake --build build
//   ./build/examples/cadence_example_loop
//
// The report prints when the loop finishes.

#include <cadence/cadence.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
    constexpr int NUM_ELEMENTS = 1 << 20;  // 2^20
    constexpr int NUM_ITERATIONS = 200;
    constexpr int NUM_THREADS_PER_BLOCK = 256;

    // basic parallelized y = ax + b
    __global__ void Saxpy(float alpha, const float* x, float* y, int numElements) {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < numElements) y[index] = alpha * x[index] + y[index];
    }

    // mults resulting array by scalar
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
    // A loop that has to close at 12.5 kHz. With no label named, the budget lands on the one scope that never launched a kernel, which is the "iteration" span below.
    config.budgetMs = 0.080;
    // The slowest iterations get written out as a timeline. Open it at https://ui.perfetto.dev to see the launches on the host lane and the kernels they queued on the device lane.
    config.tracePath = "worst.json";
    cadence::Configure(config);

    float* deviceX = nullptr;
    float* deviceY = nullptr;
    const std::size_t bytes = NUM_ELEMENTS * sizeof(float);
    if (!CudaOk(cudaMalloc(&deviceX, bytes), "cudaMalloc(x)")) return 1;
    if (!CudaOk(cudaMalloc(&deviceY, bytes), "cudaMalloc(y)")) return 1;

    cudaStream_t stream = nullptr;
    if (!CudaOk(cudaStreamCreate(&stream), "cudaStreamCreate")) return 1;

    const std::vector<float> hostX(NUM_ELEMENTS, 1.0f);
    if (!CudaOk(cudaMemcpy(deviceX, hostX.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy")) return 1;
    if (!CudaOk(cudaMemset(deviceY, 0, bytes), "cudaMemset")) return 1;

    const int numBlocks = (NUM_ELEMENTS + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

    for (int iteration = 0; iteration < NUM_ITERATIONS; ++iteration) {
        {
            CADENCE_SCOPE("iteration");

            {
                CADENCE_KERNEL("saxpy", stream);
                Saxpy<<<numBlocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(2.0f, deviceX, deviceY, NUM_ELEMENTS);
            }
            {
                CADENCE_KERNEL("scale", stream);
                Scale<<<numBlocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(0.5f, deviceY, NUM_ELEMENTS);
            }

            cudaStreamSynchronize(stream);
        }

        // Flush once per iteration
        CADENCE_FLUSH();
    }

    if (!CudaOk(cudaGetLastError(), "kernel launch")) return 1;

    CADENCE_REPORT();

    cudaStreamDestroy(stream);
    cudaFree(deviceX);
    cudaFree(deviceY);
    return 0;
}
