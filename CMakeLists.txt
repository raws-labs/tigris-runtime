cmake_minimum_required(VERSION 3.16)

option(TIGRIS_TRACE "Enable execution trace callbacks" OFF)
if(TIGRIS_TRACE)
    add_compile_definitions(TIGRIS_TRACE=1)
endif()
# A header, such as one codegen writes for a plan, that selects which operator
# kernels the runtime links. Empty links every kernel. A parent project's own
# variable wins; creating the cache entry would hide it on the first configure.
if(NOT DEFINED TIGRIS_OP_CONFIG)
    set(TIGRIS_OP_CONFIG "" CACHE FILEPATH "Header selecting the operator kernels to link")
endif()

set(TIGRIS_RUNTIME_SRCS
    src/tigris_loader.c
    src/tigris_iface.c
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
    if(TIGRIS_OP_CONFIG)
        target_compile_definitions(${COMPONENT_LIB} PRIVATE "TIGRIS_OP_CONFIG=\"${TIGRIS_OP_CONFIG}\"")
    endif()
    if(TIGRIS_TRACE)
        target_compile_definitions(${COMPONENT_LIB} PUBLIC TIGRIS_TRACE=1)
    endif()
    target_compile_options(${COMPONENT_LIB} PRIVATE -Werror=vla)
    target_link_libraries(${COMPONENT_LIB} PUBLIC m)
    return()
endif()

project(tigris_runtime VERSION 0.12.1 LANGUAGES C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(MSVC AND CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wvla)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(TIGRIS_STACK_USAGE
    "Emit compiler .su files for runtime stack-budget checks" OFF)

add_library(tigris_runtime STATIC ${TIGRIS_RUNTIME_SRCS})
add_library(tigris::runtime ALIAS tigris_runtime)
if(TIGRIS_TRACE)
    target_compile_definitions(tigris_runtime PUBLIC TIGRIS_TRACE=1)
endif()
set_target_properties(tigris_runtime PROPERTIES EXPORT_NAME runtime)
if(TIGRIS_OP_CONFIG)
    target_compile_definitions(tigris_runtime PRIVATE "TIGRIS_OP_CONFIG=\"${TIGRIS_OP_CONFIG}\"")
endif()
target_include_directories(tigris_runtime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
if(NOT WIN32)
    target_link_libraries(tigris_runtime PUBLIC m)
endif()
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

# Install a relocatable package for external CMake consumers when the runtime
# is configured directly. An embedding project owns its own install/export
# graph; registering our export there would also try to export its optional
# private backend targets (for example cmsis-nn).
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(TIGRIS_RUNTIME_CMAKE_INSTALL_DIR
        "${CMAKE_INSTALL_LIBDIR}/cmake/tigris_runtime")
    install(TARGETS tigris_runtime
        EXPORT tigris_runtimeTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h"
    )
    install(FILES LICENSE
        DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/tigris_runtime
    )
    install(EXPORT tigris_runtimeTargets
        FILE tigris_runtimeConfig.cmake
        NAMESPACE tigris::
        DESTINATION ${TIGRIS_RUNTIME_CMAKE_INSTALL_DIR}
    )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/tigris_runtimeConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        # TiGrIS is pre-1.0: keep compatibility within one minor release line.
        COMPATIBILITY SameMinorVersion
    )
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/tigris_runtimeConfigVersion.cmake"
        DESTINATION ${TIGRIS_RUNTIME_CMAKE_INSTALL_DIR}
    )
endif()

# Tests (only when built as top-level project)
option(TIGRIS_BUILD_HOST "Build the allocating host shared library and its tests" OFF)
if(TIGRIS_BUILD_HOST)
    add_subdirectory(host)
endif()

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
    target_include_directories(test_loader PRIVATE src)
    target_compile_definitions(test_loader PRIVATE
        TIGRIS_SCHEMA_COMPAT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/schema_compat")
    target_link_libraries(test_loader tigris_runtime)

    add_executable(test_executor test/test_executor.c)
    target_include_directories(test_executor PRIVATE src)
    target_link_libraries(test_executor tigris_runtime)

    # Runtime sources compiled once with TIGRIS_COUNT_KERNEL_ROWS for the two
    # row-counting twins below. Building them outside tigris_runtime keeps the
    # shipping library free of the test-only counter.
    add_library(tigris_runtime_rows_objects OBJECT ${TIGRIS_RUNTIME_SRCS})
    target_include_directories(tigris_runtime_rows_objects PRIVATE include)
    target_compile_definitions(tigris_runtime_rows_objects PRIVATE
        TIGRIS_COUNT_KERNEL_ROWS)

    # Instrumented twin: same executor tests, linked against the row-counting
    # runtime objects so the line-buffered roll tests can compare kernel
    # output-row counts.
    add_executable(test_executor_rows test/test_executor.c
        $<TARGET_OBJECTS:tigris_runtime_rows_objects>)
    target_include_directories(test_executor_rows PRIVATE include src)
    target_compile_definitions(test_executor_rows PRIVATE TIGRIS_COUNT_KERNEL_ROWS)
    target_link_libraries(test_executor_rows m)

    # The same kernel selection test against the full library and against a
    # runtime built with a selection header.
    add_executable(test_kernel_selection test/test_kernel_selection.c)
    target_link_libraries(test_kernel_selection tigris_runtime)
    add_executable(test_kernel_selection_selected
        test/test_kernel_selection.c ${TIGRIS_RUNTIME_SRCS})
    target_include_directories(test_kernel_selection_selected PRIVATE include)
    target_compile_definitions(test_kernel_selection_selected PRIVATE
        "TIGRIS_OP_CONFIG=\"${CMAKE_CURRENT_SOURCE_DIR}/test/kernel_selection_config.h\"")
    target_link_libraries(test_kernel_selection_selected m)

    # This executable references only the explicit executor API. The
    # compatibility wrapper lives in a separate archive member and must not
    # add its second workspace to the linked BSS.
    add_executable(explicit_workspace_link_smoke
        test/explicit_workspace_link_smoke.c)
    target_link_libraries(explicit_workspace_link_smoke tigris_runtime)

    add_executable(test_kernels test/test_kernels.c)
    target_include_directories(test_kernels PRIVATE src)
    target_link_libraries(test_kernels tigris_runtime)

    add_executable(test_kernels_s8 test/test_kernels_s8.c)
    target_include_directories(test_kernels_s8 PRIVATE src)
    target_link_libraries(test_kernels_s8 tigris_runtime)

    add_executable(test_lz4 test/test_lz4.c)
    target_link_libraries(test_lz4 tigris_runtime)

    add_executable(test_weight_arena test/test_weight_arena.c)
    target_include_directories(test_weight_arena PRIVATE src)
    target_link_libraries(test_weight_arena tigris_runtime)

    add_executable(test_row_offset test/test_row_offset.c)
    target_link_libraries(test_row_offset tigris_runtime)

    add_executable(test_iface test/test_iface.c)
    target_include_directories(test_iface PRIVATE src)
    target_compile_definitions(test_iface PRIVATE
        TIGRIS_SCHEMA_COMPAT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/schema_compat")
    target_link_libraries(test_iface tigris_runtime)

    add_executable(test_tile_2d test/test_tile_2d.c)
    target_include_directories(test_tile_2d PRIVATE src)
    target_compile_definitions(test_tile_2d PRIVATE
        TIGRIS_TILE2D_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/tile2d_fixtures")
    target_link_libraries(test_tile_2d tigris_runtime)

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
    # the compatibility workspace. This keeps every loader limit and the
    # fixed-workspace sizing contract live.
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

    add_executable(test_global_reduce test/test_global_reduce.c)
    target_link_libraries(test_global_reduce tigris_runtime)

    add_executable(test_tiled_transpose test/test_tiled_transpose.c)
    target_link_libraries(test_tiled_transpose tigris_runtime)

    add_executable(test_tiled_rows test/test_tiled_rows.c)
    target_link_libraries(test_tiled_rows tigris_runtime)

    # The storage test against the default runtime and against one built with
    # every limit at the plan format's maximum.
    find_package(Threads REQUIRED)
    add_executable(test_plan_storage test/test_plan_storage.c)
    target_include_directories(test_plan_storage PRIVATE src)
    target_link_libraries(test_plan_storage tigris_runtime Threads::Threads)
    set(TIGRIS_FORMAT_MAXIMA
        TIGRIS_MAX_TENSORS=65535u TIGRIS_MAX_STAGE_INPUTS=65535u
        TIGRIS_MAX_STAGE_OUTPUTS=65535u TIGRIS_MAX_CHAIN_STAGES=65535u
        TIGRIS_MAX_SUBGRAPHS=65535u TIGRIS_MAX_SUBGRAPH_DEPTH=65535u
        TIGRIS_MAX_SPATIAL_OPS_PER_STAGE=65535u)
    add_library(tigris_runtime_maxima_objects OBJECT ${TIGRIS_RUNTIME_SRCS})
    target_include_directories(tigris_runtime_maxima_objects PRIVATE include)
    target_compile_definitions(tigris_runtime_maxima_objects PRIVATE ${TIGRIS_FORMAT_MAXIMA})
    add_executable(test_plan_storage_maxima test/test_plan_storage.c
        $<TARGET_OBJECTS:tigris_runtime_maxima_objects>)
    target_include_directories(test_plan_storage_maxima PRIVATE include src)
    target_compile_definitions(test_plan_storage_maxima PRIVATE ${TIGRIS_FORMAT_MAXIMA})
    target_link_libraries(test_plan_storage_maxima m Threads::Threads)

    add_executable(test_avg_pool_s8 test/test_avg_pool_s8.c)
    target_include_directories(test_avg_pool_s8 PRIVATE src)
    target_link_libraries(test_avg_pool_s8 tigris_runtime)

    # Host-only bridge used by the compiler repository's cross-contract gate.
    # It is deliberately not a CTest case: its plan, input, and expected-output
    # files are generated by that gate.
    add_executable(tigris_contract_runner test/contract_runner.c)
    target_link_libraries(tigris_contract_runner tigris_runtime)

    # Rows-instrumented twin of the contract runner, linked against the
    # row-counting runtime objects so the gate can read kernel output-row
    # counts and prove the line-buffered roll computes fewer rows than the
    # recompute path. Also not a CTest case, for the same reason as the
    # default runner.
    add_executable(tigris_contract_runner_rows
        test/contract_runner.c $<TARGET_OBJECTS:tigris_runtime_rows_objects>)
    target_include_directories(tigris_contract_runner_rows PRIVATE include)
    target_compile_definitions(tigris_contract_runner_rows PRIVATE
        TIGRIS_COUNT_KERNEL_ROWS)
    target_link_libraries(tigris_contract_runner_rows m)

    # Self-contained unit tests.
    add_test(NAME loader COMMAND test_loader)
    add_test(NAME executor COMMAND test_executor)
    add_test(NAME executor_rows COMMAND test_executor_rows)
    add_test(NAME kernel_selection COMMAND test_kernel_selection)
    add_test(NAME kernel_selection_selected COMMAND test_kernel_selection_selected)
    add_test(NAME explicit_workspace_link COMMAND explicit_workspace_link_smoke)
    add_test(NAME kernels_f32 COMMAND test_kernels)
    add_test(NAME kernels_s8 COMMAND test_kernels_s8)
    add_test(NAME lz4 COMMAND test_lz4)
    add_test(NAME weight_arena COMMAND test_weight_arena)
    add_test(NAME row_offset COMMAND test_row_offset)
    add_test(NAME tile_2d COMMAND test_tile_2d)
    add_test(NAME iface COMMAND test_iface)
    add_test(NAME accel_routing COMMAND test_accel_routing)
    add_test(NAME cmsis_arena COMMAND test_cmsis_arena)
    add_test(NAME global_reduce COMMAND test_global_reduce)
    add_test(NAME tiled_transpose COMMAND test_tiled_transpose)
    add_test(NAME tiled_rows COMMAND test_tiled_rows)
    add_test(NAME avg_pool_s8 COMMAND test_avg_pool_s8)
    add_test(NAME plan_storage COMMAND test_plan_storage)
    add_test(NAME plan_storage_maxima COMMAND test_plan_storage_maxima)
    add_test(NAME posix_example COMMAND tigris_posix_example
        ${CMAKE_CURRENT_SOURCE_DIR}/test/schema_compat/schema-v9-binary-requant.tgrs)
    # The output pattern replaces the exit-code check, so sanitizer reports fail it explicitly.
    set_tests_properties(posix_example PROPERTIES
        PASS_REGULAR_EXPRESSION "Output: 0\\.18 0\\.18 0\\.18 0\\.18 0\\.18\nFast arena peak: 768 bytes"
        FAIL_REGULAR_EXPRESSION "Sanitizer;runtime error")
    add_test(NAME posix_example_int8_output COMMAND tigris_posix_example
        ${CMAKE_CURRENT_SOURCE_DIR}/test/schema_compat/schema-v11-scheduled-peak.tgrs)
    set_tests_properties(posix_example_int8_output PROPERTIES
        PASS_REGULAR_EXPRESSION "Output: -1 1 -14 -5 -1\n"
        FAIL_REGULAR_EXPRESSION "Sanitizer;runtime error")
endif()
