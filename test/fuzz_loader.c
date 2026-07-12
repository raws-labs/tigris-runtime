/**
 * @file fuzz_loader.c
 * @brief LibFuzzer entry point for the zero-allocation plan loader.
 */

#include "tigris_loader.h"

#include <stddef.h>
#include <stdint.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    tigris_plan_t plan;

    if (size > UINT32_MAX)
        return 0;

    (void)tigris_plan_load(data, (uint32_t)size, &plan);
    return 0;
}
