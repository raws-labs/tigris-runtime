cmake_minimum_required(VERSION 3.16)
project(tigris_runtime C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic -Werror)

add_library(tigris_runtime STATIC
    src/tigris_loader.c
    src/tigris_mem.c
    src/tigris_executor.c
    src/tigris_kernels.c
    src/tigris_kernels_s8.c
    src/tigris_kernels_esp_nn.c
    src/tigris_kernels_cmsis_nn.c
    src/tigris_lz4.c
)
target_include_directories(tigris_runtime PUBLIC include)
target_link_libraries(tigris_runtime PUBLIC m)

# Optional: ESP-NN backend (define TIGRIS_HAS_ESP_NN and link esp-nn)
# target_compile_definitions(tigris_runtime PUBLIC TIGRIS_HAS_ESP_NN)
# target_link_libraries(tigris_runtime PUBLIC esp-nn)

# Optional: CMSIS-NN backend (define TIGRIS_HAS_CMSIS_NN and link cmsis-nn)
# target_compile_definitions(tigris_runtime PUBLIC TIGRIS_HAS_CMSIS_NN)
# target_link_libraries(tigris_runtime PUBLIC cmsis-nn)

# Tests (only when built as top-level project)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    enable_testing()

    add_executable(test_loader test/test_loader.c)
    target_link_libraries(test_loader tigris_runtime)

    add_executable(test_executor test/test_executor.c)
    target_link_libraries(test_executor tigris_runtime)

    add_executable(test_kernels test/test_kernels.c)
    target_link_libraries(test_kernels tigris_runtime)

    add_executable(test_e2e test/test_e2e.c)
    target_link_libraries(test_e2e tigris_runtime)

    add_executable(test_kernels_s8 test/test_kernels_s8.c)
    target_link_libraries(test_kernels_s8 tigris_runtime)

    add_executable(test_lz4 test/test_lz4.c)
    target_link_libraries(test_lz4 tigris_runtime)

    add_executable(test_s8_e2e test/test_s8_e2e.c)
    target_link_libraries(test_s8_e2e tigris_runtime)

    add_executable(dump_plan test/dump_plan.c)
    target_link_libraries(dump_plan tigris_runtime)

    add_executable(test_yolo_s8 test/test_yolo_s8.c)
    target_link_libraries(test_yolo_s8 tigris_runtime)

    add_executable(test_plan_contract test/test_plan_contract.c)
    target_link_libraries(test_plan_contract tigris_runtime)
endif()
