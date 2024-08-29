#ifndef INCLUDE_CK_TILE_OPS_FMHA_PRINT_UTILS_H_
#define INCLUDE_CK_TILE_OPS_FMHA_PRINT_UTILS_H_

#define PRINT_ONLY_IN_GRID(...)                                \
    if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) \
    {                                                          \
        printf("LMS: batch[%d]: \n", blockIdx.z);              \
        printf(__VA_ARGS__);                                   \
    }

// #define PRINT_ONLY_IN_GRID(...)

#endif // INCLUDE_CK_TILE_OPS_FMHA_PRINT_UTILS_H_
