#include <stdio.h>

#define log(...)                                         \
    ({                                                   \
        fprintf(stderr, "[%s:%u] ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                    \
        fprintf(stderr, "\n");                           \
    })
