/** Link-only regression for explicit-workspace applications. */

#include <string.h>

#include "tigris_executor.h"

static tigris_executor_workspace_t tigris_explicit_executor_workspace;

int main(void)
{
    tigris_plan_t plan;
    tigris_mem_t mem;
    memset(&plan, 0, sizeof(plan));
    memset(&mem, 0, sizeof(mem));

    return tigris_run_with_workspace(
               &plan, &mem, NULL, NULL, NULL,
               &tigris_explicit_executor_workspace) == TIGRIS_EXEC_ERR_NULL
        ? 0 : 1;
}
