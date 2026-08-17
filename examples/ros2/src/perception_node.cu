// A ROS 2 node whose timer callback runs a small CUDA pipeline, instrumented with cadence.
//
// The question a ROS 2 developer actually has about a perception callback is not "how long does it take on average" but "is it holding its rate, and when it does not, which stage was it". A topic carrying one latency number per callback answers neither: it tells you the callback was slow without telling you where the time went, and by the time you plot it the run is over.
//
// So this node does both. It publishes its own end-to-end latency on ~/latency_ms, which is what you would write anyway, and it wraps each stage in a cadence scope. The deadline is set to the timer period, so the report's verdict is the question the node exists to answer: did this callback hold its rate. When it did not, the slowest-iteration breakdown says which stage cost the time.
//
// Nothing here is synthetic-slowdown theatre. At the default period the pipeline fits comfortably; drop the period or raise the workload and the misses are real ones caused by real work.

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

    // Three stages standing in for a perception pipeline. They are arithmetic rather than a real detector, but they are real GPU work with real launch costs, which is what the instrumentation is being shown against.
    __global__ void Normalize(float* data, int numElements, float scale, float bias) {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= numElements) return;
        data[index] = fmaf(data[index], scale, bias);
    }

    // The heavy stage, and the one whose cost moves when the workload parameter changes.
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
            // Defaults chosen so the pipeline uses a visible fraction of its period rather than a rounding error of it. A demo whose deadline bar sits at 2% teaches nothing: the interesting output is a loop that is comfortably meeting a rate you can then take away from it.
            periodMs_ = declare_parameter<double>("period_ms", 5.0);
            numElements_ = static_cast<int>(declare_parameter<int>("num_elements", 1 << 20));
            numTaps_ = static_cast<int>(declare_parameter<int>("num_taps", 384));

            // The deadline is the timer period, which is what makes the report's verdict answer the node's actual question rather than a generic one. No budgetLabel is named: the sole label that records host time and launches nothing is the CADENCE_SCOPE around the callback, and cadence resolves the budget to it on its own.
            cadence::Config config;
            config.budgetMs = periodMs_;
            // The first callbacks pay for context creation and module loading, which have nothing to do with whether the loop holds its rate.
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
                // The loop span. Everything below is inside it, so the deadline is held against the whole callback rather than against any one stage.
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

                // The boundary this callback was going to synchronize on anyway, which is exactly where cadence wants its flush: the records resolve against work that has already finished, so nothing is waited on twice.
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
    // Explicit, and after the node is gone, so the report is taken while the CUDA runtime is certainly still alive rather than during static destruction.
    CADENCE_REPORT();
    return 0;
}
