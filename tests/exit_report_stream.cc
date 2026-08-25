// Exit-time reporting must never dereference a stream cadence does not own.
//
// The atexit handler runs after main returns, by which point a caller's stream
// may be long destroyed. Observing what the handler does therefore needs a
// whole process, which is why this is a separate program rather than a case in
// host_tests.cc.
//
//   exit_report_stream dangling   reportStream is dead by exit; expect a notice
//   exit_report_stream standard   reportStream is std::cout; expect the report

#include <cadence/cadence.h>

#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "standard";

    cadence::Config config;
    config.warmupIterations = 0;

    if (mode == "dangling") {
        // A stream the caller owns and destroys, which is what a logger torn down
        // during shutdown looks like. The registry keeps the pointer either way.
        //
        // Heap rather than stack on purpose. Writing through a destroyed stack
        // stream after main returns is silent -- no crash, no sanitizer report,
        // just a report that vanishes -- so only the heap version fails loudly
        // enough for the ASan job to catch a regression here.
        std::ostringstream* sink = new std::ostringstream();
        config.reportStream = sink;
        cadence::Configure(config);
        cadence::detail::Registry::Instance().RecordHost("work", 1.0);
        delete sink;
    } else {
        config.reportStream = &std::cout;
        cadence::Configure(config);
        cadence::detail::Registry::Instance().RecordHost("work", 1.0);
    }

    // Deliberately no cadence::Report(). The exit-time fallback is the thing
    // under test, and calling Report() would suppress it.
    return 0;
}
