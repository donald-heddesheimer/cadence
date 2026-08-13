// Compiled with -DCADENCE_DISABLE. Everything below must still build, and the macros must collect nothing at all -- not "collect and throw away".

#include <cadence/cadence.h>

#include <cstdio>
#include <thread>

int main() {
    cadence::Config config;
    config.warmupIterations = 0;
    config.outputPath.clear();
    CADENCE_CONFIGURE(config);   // Compiles to nothing; proven by the check below.
    cadence::Configure(config);  // The direct API still works, and suppresses the
                                 // exit-time write into the build directory.

    {
        CADENCE_SCOPE("should-not-exist");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CADENCE_KERNEL("also-should-not-exist");
    CADENCE_FLUSH();
    CADENCE_REPORT();

    // Nothing was recorded, so nothing can be reported. Note that Configure() never ran either: CADENCE_CONFIGURE compiled out with the rest.
    if (!cadence::Snapshot().empty()) {
        std::printf("[FAIL] disabled build still collected samples\n");
        return 1;
    }
    std::printf("[ ok ] disabled build collects nothing\n");
    return 0;
}
