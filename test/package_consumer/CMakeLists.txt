cmake_minimum_required(VERSION 3.16)

project(tigris_runtime_package_consumer LANGUAGES C)

find_package(tigris_runtime 0.5 REQUIRED CONFIG)

add_executable(tigris_runtime_package_consumer main.c)
set_target_properties(tigris_runtime_package_consumer PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
)
target_link_libraries(tigris_runtime_package_consumer PRIVATE tigris::runtime)

enable_testing()
add_test(
    NAME installed_package_consumer
    COMMAND tigris_runtime_package_consumer
)
