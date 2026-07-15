cmake_minimum_required(VERSION 3.16)

set(TIGRIS_RUNTIME_SRCS
    src/tigris_loader.c
    src/tigris_mem.c
    src/tigris_executor.c
    src/tigris_executor_compat.c
    src/tigris_kernels.c
    src/tigris_kernels_s8.c
    src/tigris_kernels_esp_nn.c
    src/tigris_kernels_cmsis_nn.c
    src/tigris_lz4.c
)

if(COMMAND idf_component_register)
    idf_component_register(
        SRCS ${TIGRIS_RUNTIME_SRCS}
        INCLUDE_DIRS include
        REQUIRES esp-nn
    )
    target_compile_definitions(${COMPONENT_LIB} PUBLIC TIGRIS_HAS_ESP_NN)
    target_compile_options(${COMPONENT_LIB} PRIVATE -Werror=vla)
    target_link_libraries(${COMPONENT_LIB} PUBLIC m)
    return()
endif()

project(tigris_runtime C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wvla)

option(TIGRIS_STACK_USAGE
    "Emit compiler .su files for runtime stack-budget checks" OFF)

add_library(tigris_runtime STATIC ${TIGRIS_RUNTIME_SRCS})
target_include_directories(tigris_runtime PUBLIC include)
target_link_libraries(tigris_runtime PUBLIC m)
if(TIGRIS_STACK_USAGE)
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_compile_options(tigris_runtime PRIVATE -fstack-usage)
    else()
        message(FATAL_ERROR "TIGRIS_STACK_USAGE requires GCC")
    endif()
endif()

# Optional: ESP-NN backend (define TIGRIS_HAS_ESP_NN and link esp-nn)
# target_compile_definitions(tigris_runtime PUBLIC TIGRIS_HAS_ESP_NN)
# target_link_libraries(tigris_runtime PUBLIC esp-nn)

# Optional: CMSIS-NN backend for Arm Cortex-M.
# Enable with -DTIGRIS_HAS_CMSIS_NN=ON. The including project (or the link line)
# must provide a `cmsis-nn` target: the ARM CMSIS-NN static library plus its
# CMSIS-Core / CMSIS-DSP include directories (added as SYSTEM includes so the
# runtime's -Wpedantic -Werror does not trip on the vendor headers).
option(TIGRIS_HAS_CMSIS_NN "Enable the CMSIS-NN int8 kernel backend" OFF)
if(TIGRIS_HAS_CMSIS_NN)
    target_compile_definitions(tigris_runtime PUBLIC TIGRIS_HAS_CMSIS_NN)
    target_link_libraries(tigris_runtime PUBLIC cmsis-nn)
endif()

# Tests (only when built as top-level project)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    enable_testing()

    # Canonical integration example. Keep its application code valid C99
    # even though the library's top-level developer build uses C11.
    add_executable(tigris_posix_example examples/posix/main.c)
    set_target_properties(tigris_posix_example PROPERTIES
        C_STANDARD 99
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF
    )
    target_link_libraries(tigris_posix_example tigris_runtime)

    add_executable(test_loader test/test_loader.c)
    target_link_libraries(test_loader tigris_runtime)

    add_executable(test_executor test/test_executor.c)
    target_link_libraries(test_executor tigris_runtime)

    # This executable references only the explicit executor API. The legacy
    # wrapper lives in a separate archive member and must not add its second
    # workspace to the linked BSS.
    add_executable(explicit_workspace_link_smoke
        test/explicit_workspace_link_smoke.c)
    target_link_libraries(explicit_workspace_link_smoke tigris_runtime)

    add_executable(test_kernels test/test_kernels.c)
    target_link_libraries(test_kernels tigris_runtime)

    add_executable(test_e2e test/test_e2e.c)
    target_link_libraries(test_e2e tigris_runtime)

    add_executable(test_kernels_s8 test/test_kernels_s8.c)
    target_link_libraries(test_kernels_s8 tigris_runtime)

    add_executable(test_lz4 test/test_lz4.c)
    target_link_libraries(test_lz4 tigris_runtime)

    add_executable(test_weight_arena test/test_weight_arena.c)
    target_link_libraries(test_weight_arena tigris_runtime)

    add_executable(test_tile_padding test/test_tile_padding.c)
    target_link_libraries(test_tile_padding tigris_runtime)

    add_executable(test_accel_routing test/test_accel_routing.c)
    target_include_directories(test_accel_routing PRIVATE src)
    target_link_libraries(test_accel_routing tigris_runtime)

    # Compile the CMSIS adapter in native CI even when the vendor library is
    # unavailable. The stub supplies declarations only; this target catches
    # API/type/warning regressions in code hidden by TIGRIS_HAS_CMSIS_NN.
    add_library(cmsis_adapter_compile OBJECT src/tigris_kernels_cmsis_nn.c)
    target_compile_definitions(cmsis_adapter_compile PRIVATE TIGRIS_HAS_CMSIS_NN)
    target_include_directories(cmsis_adapter_compile PRIVATE
        include src test/cmsis_stub)
    if(TIGRIS_STACK_USAGE)
        target_compile_options(cmsis_adapter_compile PRIVATE -fstack-usage)
    endif()

    # Compile the executor and loader with reduced public limits and without
    # the compatibility workspace. This keeps all configuration knobs live
    # and verifies that the opaque workspace formula still bounds its private
    # implementation when integrators tune for a smaller target.
    add_library(stack_config_compile OBJECT
        src/tigris_executor.c src/tigris_executor_compat.c
        src/tigris_loader.c)
    target_include_directories(stack_config_compile PRIVATE include)
    target_compile_definitions(stack_config_compile PRIVATE
        TIGRIS_MAX_TENSORS=8
        TIGRIS_MAX_STAGE_INPUTS=2
        TIGRIS_MAX_STAGE_OUTPUTS=2
        TIGRIS_MAX_CHAIN_STAGES=3
        TIGRIS_MAX_SPATIAL_OPS_PER_STAGE=2
        TIGRIS_ENABLE_DEFAULT_EXECUTOR_WORKSPACE=0)

    add_executable(test_cmsis_arena
        test/test_cmsis_arena.c $<TARGET_OBJECTS:cmsis_adapter_compile>)
    target_compile_definitions(test_cmsis_arena PRIVATE TIGRIS_HAS_CMSIS_NN)
    target_include_directories(test_cmsis_arena PRIVATE test/cmsis_stub)
    target_link_libraries(test_cmsis_arena tigris_runtime)

    add_executable(test_avg_pool_s8 test/test_avg_pool_s8.c)
    target_include_directories(test_avg_pool_s8 PRIVATE src)
    target_link_libraries(test_avg_pool_s8 tigris_runtime)

    add_executable(test_s8_e2e test/test_s8_e2e.c)
    target_link_libraries(test_s8_e2e tigris_runtime)

    add_executable(dump_plan test/dump_plan.c)
    target_link_libraries(dump_plan tigris_runtime)

    add_executable(test_yolo_s8 test/test_yolo_s8.c)
    target_link_libraries(test_yolo_s8 tigris_runtime)

    add_executable(test_plan_contract test/test_plan_contract.c)
    target_link_libraries(test_plan_contract tigris_runtime)

    # Host-only bridge used by the compiler repository's cross-contract gate.
    # It is deliberately not a CTest case: its plan, input, and expected-output
    # files are generated by that gate.
    add_executable(tigris_contract_runner test/contract_runner.c)
    target_link_libraries(tigris_contract_runner tigris_runtime)

    # Self-contained unit tests.  Fixture-driven executables are built above,
    # but are not registered here because a clean runtime checkout does not
    # contain the generated .tgrs model fixtures they require.
    add_test(NAME loader COMMAND test_loader)
    add_test(NAME executor COMMAND test_executor)
    add_test(NAME explicit_workspace_link COMMAND explicit_workspace_link_smoke)
    add_test(NAME kernels_f32 COMMAND test_kernels)
    add_test(NAME kernels_s8 COMMAND test_kernels_s8)
    add_test(NAME lz4 COMMAND test_lz4)
    add_test(NAME weight_arena COMMAND test_weight_arena)
    add_test(NAME tile_padding COMMAND test_tile_padding)
    add_test(NAME accel_routing COMMAND test_accel_routing)
    add_test(NAME cmsis_arena COMMAND test_cmsis_arena)
    add_test(NAME avg_pool_s8 COMMAND test_avg_pool_s8)
endif()
