// The macros intern once per call site, so a label computed at runtime would file
// every scope under its first value. This must not compile.
//
// CTest builds it twice: once with CADENCE_TRY_RUNTIME_LABEL, expecting failure,
// and once without, expecting success. The control matters -- without it, a typo
// here would make the negative test pass for the wrong reason.

#include <cadence/cadence.h>

const char* RuntimeLabel();

int main() {
    CADENCE_SCOPE("literal");
#ifdef CADENCE_TRY_RUNTIME_LABEL
    CADENCE_SCOPE(RuntimeLabel());
#endif
    return 0;
}
