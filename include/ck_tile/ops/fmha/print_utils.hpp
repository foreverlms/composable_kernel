#ifndef INCLUDE_CK_TILE_OPS_FMHA_PRINT_UTILS_H_
#define INCLUDE_CK_TILE_OPS_FMHA_PRINT_UTILS_H_

#define PRINT_ONLY_IN_GRID(...)                                                                    \
    if(threadIdx.x == 0)                                                                           \
    {                                                                                              \
        printf(                                                                                    \
            "LMS: batch[%d], head*splits[%d], q_tile[%d]:\n", blockIdx.z, blockIdx.y, blockIdx.x); \
        printf(__VA_ARGS__);                                                                       \
    }

// #define PRINT_ONLY_IN_GRID(...)

#endif // INCLUDE_CK_TILE_OPS_FMHA_PRINT_UTILS_H_
