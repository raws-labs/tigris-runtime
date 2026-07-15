/**
 * @file tigris_executor_compat.c
 * @brief Source-compatible executor wrapper with process-global workspace.
 *
 * This wrapper intentionally lives in its own translation unit. Static-library
 * consumers of tigris_run_with_workspace() therefore do not pull the legacy
 * workspace into their linked BSS.
 */

#include "tigris_executor.h"

#if TIGRIS_ENABLE_DEFAULT_EXECUTOR_WORKSPACE
static tigris_executor_workspace_t tigris_default_executor_workspace;
#endif

tigris_exec_error_t tigris_run(
    const tigris_plan_t *plan,
    tigris_mem_t        *mem,
    tigris_kernel_fn     kernel,
    void                *user_ctx,
    tigris_exec_stats_t *stats)
{
#if TIGRIS_ENABLE_DEFAULT_EXECUTOR_WORKSPACE
    return tigris_run_with_workspace(
        plan, mem, kernel, user_ctx, stats,
        &tigris_default_executor_workspace);
#else
    (void)plan;
    (void)mem;
    (void)kernel;
    (void)user_ctx;
    (void)stats;
    return TIGRIS_EXEC_ERR_WORKSPACE;
#endif
}
