#ifndef TIGRIS_HOST_H
#define TIGRIS_HOST_H

#include <stdint.h>

#if defined(_WIN32)
# if defined(TIGRIS_HOST_BUILD)
#  define TIGRIS_HOST_API __declspec(dllexport)
# else
#  define TIGRIS_HOST_API __declspec(dllimport)
# endif
#else
# define TIGRIS_HOST_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tigris_host tigris_host_t;

TIGRIS_HOST_API uint32_t tigris_host_abi(void);
TIGRIS_HOST_API const char *tigris_host_version(void);

/* Calls return NULL on success or a static error string. Creation copies the
 * plan and allocates all storage. A zero slow capacity selects a conservative
 * host allocation; it is not a compiled slow-memory requirement. */
TIGRIS_HOST_API const char *tigris_host_create(
    const void *data, uint32_t size, uint32_t slow_capacity, tigris_host_t **out);
TIGRIS_HOST_API void tigris_host_destroy(tigris_host_t *host);

/* Tensors use stored axis order and declared interface dtype. output is 0 for
 * inputs and 1 for outputs. Invalid queries return zero or NULL. */
TIGRIS_HOST_API uint32_t tigris_host_tensor_count(const tigris_host_t *host, int output);
TIGRIS_HOST_API const char *tigris_host_tensor_name(const tigris_host_t *host, int output, uint32_t index);
TIGRIS_HOST_API uint32_t tigris_host_tensor_bytes(const tigris_host_t *host, int output, uint32_t index);
TIGRIS_HOST_API uint32_t tigris_host_tensor_dtype(const tigris_host_t *host, int output, uint32_t index);
TIGRIS_HOST_API uint32_t tigris_host_tensor_rank(const tigris_host_t *host, int output, uint32_t index);
TIGRIS_HOST_API int32_t tigris_host_tensor_dim(const tigris_host_t *host, int output, uint32_t index, uint32_t axis);

/* Every run resets the arenas. Distinct sessions can execute concurrently;
 * callers must serialize access to a single session. No allocation occurs here. */
TIGRIS_HOST_API const char *tigris_host_run(
    tigris_host_t *host,
    const void *const *inputs, const uint32_t *input_sizes, uint32_t input_count,
    void *const *outputs, const uint32_t *output_sizes, uint32_t output_count);

enum {
    TIGRIS_HOST_FAST_CAPACITY = 0, TIGRIS_HOST_SLOW_CAPACITY = 1,
    TIGRIS_HOST_FAST_PEAK = 2, TIGRIS_HOST_SLOW_PEAK = 3,
    TIGRIS_HOST_WORKSPACE_BYTES = 4
};
/* Peaks cover execution arenas, not process memory. */
TIGRIS_HOST_API uint64_t tigris_host_metric(const tigris_host_t *host, uint32_t metric);

#ifdef __cplusplus
}
#endif
#endif
