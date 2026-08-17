// A ROS 2 timer callback instrumented for end-to-end and per-stage CUDA latency.

#include <cadence/cadence.h>

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"

namespace {

    // Three arithmetic stages model the launch and execution shape of a pipeline.
    __global__ void Normalize(float* data, int numElements, float scale, float bias) {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numElements) return;
        data[index] = fmaf(data[index], scale, bias);
    }

    // This stage scales with the configurable workload.
    __global__ void Detect(const float* data, float* scores, int numElements, int numTaps) {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numElements) return;
        float accumulated = 0.0f;
        for (int tap = 0; tap < numTaps; ++tap) {
            const int sample = (index + tap * 37) % numElements;
            accumulated = fmaf(data[sample], 1.0f / static_cast<float>(tap + 1), accumulated);
        }
        scores[index] = accumulated;
    }

    __global__ void Threshold(const float* scores, float* output, int numElements, float cutoff) {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numElements) return;
        output[index] = scores[index] > cutoff ? scores[index] : 0.0f;
    }

    constexpr int NUM_THREADS_PER_BLOCK = 256;

    int NumBlocksFor(int numElements) {
        return (numElements + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    }

    class PerceptionNode : public rclcpp::Node {
       public:
        PerceptionNode() : Node("cadence_perception") {
            // Defaults keep the workload below, but close enough to, its deadline.
            periodMs_ = declare_parameter<double>("period_ms", 5.0);
            numElements_ = static_cast<int>(declare_parameter<int>("num_elements", 1 << 20));
            numTaps_ = static_cast<int>(declare_parameter<int>("num_taps", 384));

            // Apply the timer period to the automatically selected callback scope.
            cadence::Config config;
            config.budgetMs = periodMs_;
            // Exclude context creation and module loading from steady-state results.
            config.warmupIterations = 10;
            config.numWorstIterations = 3;
            config.tracePath = "cadence_perception.json";
            cadence::Configure(config);

            AllocateOrThrow();
            cudaStreamCreate(&stream_);

            publisher_ = create_publisher<std_msgs::msg::Float32>("~/latency_ms", 10);
            timer_ = create_wall_timer(std::chrono::duration<double, std::milli>(periodMs_),
                                       [this]() { OnTimer(); });

            RCLCPP_INFO(get_logger(), "publishing every %.1f ms; %d elements, %d taps; deadline %.1f ms",
                        periodMs_, numElements_, numTaps_, periodMs_);
        }

        ~PerceptionNode() override {
            if (stream_) cudaStreamDestroy(stream_);
            cudaFree(input_);
            cudaFree(scores_);
            cudaFree(output_);
        }

       private:
        void AllocateOrThrow() {
            const std::size_t bytes = static_cast<std::size_t>(numElements_) * sizeof(float);
            if (cudaMalloc(&input_, bytes) != cudaSuccess || cudaMalloc(&scores_, bytes) != cudaSuccess ||
                cudaMalloc(&output_, bytes) != cudaSuccess) {
                throw std::runtime_error("could not allocate device buffers");
            }
            const std::vector<float> seed(static_cast<std::size_t>(numElements_), 1.0f);
            cudaMemcpy(input_, seed.data(), bytes, cudaMemcpyHostToDevice);
        }

        void OnTimer() {
            const auto began = std::chrono::steady_clock::now();
            {
                // Measure the complete callback for deadline evaluation.
                CADENCE_SCOPE("callback");

                const int numBlocks = NumBlocksFor(numElements_);
                {
                    CADENCE_KERNEL("normalize", stream_);
                    Normalize<<<numBlocks, NUM_THREADS_PER_BLOCK, 0, stream_>>>(input_, numElements_, 0.999f, 1e-4f);
                }
                {
                    CADENCE_KERNEL("detect", stream_);
                    Detect<<<numBlocks, NUM_THREADS_PER_BLOCK, 0, stream_>>>(input_, scores_, numElements_, numTaps_);
                }
                {
                    CADENCE_KERNEL("threshold", stream_);
                    Threshold<<<numBlocks, NUM_THREADS_PER_BLOCK, 0, stream_>>>(scores_, output_, numElements_, 0.5f);
                }

                // Resolve records at the callback's existing synchronization boundary.
                cudaStreamSynchronize(stream_);
                CADENCE_FLUSH();
            }

            const double elapsedMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count();
            std_msgs::msg::Float32 message;
            message.data = static_cast<float>(elapsedMs);
            publisher_->publish(message);
        }

        double periodMs_ = 20.0;
        int numElements_ = 0;
        int numTaps_ = 0;
        float* input_ = nullptr;
        float* scores_ = nullptr;
        float* output_ = nullptr;
        cudaStream_t stream_ = nullptr;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_;
        rclcpp::TimerBase::SharedPtr timer_;
    };

}  // namespace

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    {
        auto node = std::make_shared<PerceptionNode>();
        rclcpp::spin(node);
    }
    rclcpp::shutdown();
    // Report before CUDA runtime teardown.
    CADENCE_REPORT();
    return 0;
}
