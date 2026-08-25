// The macros intern once per call site, so a label whose characters can change
// would file every scope under its first value. Only a const character array is
// accepted: its contents cannot be rewritten after the handle is cached.
//
// CTest builds this three times: once clean, expecting success, and once for
// each rejection, expecting failure. The clean build is the control -- without
// it, a typo here would make the negative tests pass for the wrong reason.
//
// Both rejected cases live at namespace scope on purpose. CADENCE_DETAIL_LABEL
// wraps the intern in a captureless lambda, so any local is already refused
// whatever its type, and testing one would prove nothing about the check.

#include <cadence/cadence.h>

const char* RuntimeLabel();

// Stable: contents are fixed for the life of the program.
const char kConstArrayLabel[] = "const-array";

// Not stable: nothing stops this being rewritten after the first scope caches it.
char gMutableArrayLabel[32] = "mutable-array";

int main() {
    CADENCE_SCOPE("literal");
    CADENCE_SCOPE(kConstArrayLabel);
#ifdef CADENCE_TRY_RUNTIME_LABEL
    CADENCE_SCOPE(RuntimeLabel());
#endif
#ifdef CADENCE_TRY_MUTABLE_LABEL
    CADENCE_SCOPE(gMutableArrayLabel);
#endif
    return 0;
}
