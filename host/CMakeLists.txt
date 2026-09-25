add_library(tigris_host SHARED tigris_host.c)
foreach(source IN LISTS TIGRIS_RUNTIME_SRCS)
    target_sources(tigris_host PRIVATE "${PROJECT_SOURCE_DIR}/${source}")
endforeach()
target_include_directories(tigris_host PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(tigris_host PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(tigris_host PRIVATE
    TIGRIS_HOST_BUILD TIGRIS_HOST_VERSION="${PROJECT_VERSION}"
    TIGRIS_ENABLE_DEFAULT_EXECUTOR_WORKSPACE=0 TIGRIS_TENSOR_ALIGN=32)
set_target_properties(tigris_host PROPERTIES
    C_VISIBILITY_PRESET hidden
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
if(NOT WIN32)
    target_link_libraries(tigris_host PRIVATE m)
endif()

add_executable(test_host test_host.c)
target_link_libraries(test_host PRIVATE tigris_host)
target_compile_definitions(test_host PRIVATE
    TIGRIS_SCHEMA_COMPAT_DIR="${PROJECT_SOURCE_DIR}/test/schema_compat")
enable_testing()
add_test(NAME host_library COMMAND test_host)

set(TIGRIS_HOST_PLATFORM "" CACHE STRING "Distribution platform tag for the host archive")
if(NOT TIGRIS_HOST_PLATFORM)
    string(TOLOWER "${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}" TIGRIS_HOST_PLATFORM)
endif()
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE TIGRIS_HOST_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE)
add_custom_target(host_archive
    COMMAND "${CMAKE_COMMAND}"
        "-DLIBRARY=$<TARGET_FILE:tigris_host>"
        "-DHEADER=${CMAKE_CURRENT_SOURCE_DIR}/tigris_host.h"
        "-DLICENSE=${PROJECT_SOURCE_DIR}/LICENSE"
        "-DVERSION=${PROJECT_VERSION}"
        "-DREVISION=${TIGRIS_HOST_REVISION}"
        "-DPLATFORM=${TIGRIS_HOST_PLATFORM}"
        "-DOUTPUT=${CMAKE_BINARY_DIR}/host-artifacts"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/package.cmake"
    DEPENDS tigris_host)
