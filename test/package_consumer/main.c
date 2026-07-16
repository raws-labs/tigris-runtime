#include <stddef.h>

#include "tigris_executor.h"
#include "tigris_loader.h"

int main(void)
{
    const char *error = tigris_error_str(TIGRIS_ERR_NULL);
    if (error == NULL) {
        return 1;
    }

    return tigris_executor_workspace_size() ==
                   sizeof(tigris_executor_workspace_t)
        ? 0
        : 1;
}
