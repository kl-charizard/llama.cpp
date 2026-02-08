#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

struct ggml_awq_tensor_extra {
    uint32_t n;
    uint32_t n_expert;
    const uint32_t * idx;
    const struct ggml_tensor * weights;
};

#define GGML_AWQ_TENSOR_SUFFIX ".awq_f16"

#ifdef __cplusplus
}
#endif
