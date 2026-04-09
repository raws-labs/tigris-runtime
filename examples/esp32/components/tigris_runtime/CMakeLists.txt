if(NOT DEFINED TIGRIS_RUNTIME_DIR)
    # Default: this example lives inside tigris-runtime/examples/esp32/
    set(TIGRIS_RUNTIME_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../.."
        CACHE PATH "Path to tigris-runtime root")
endif()

if(NOT EXISTS "${TIGRIS_RUNTIME_DIR}/include/tigris.h")
    message(FATAL_ERROR
        "tigris-runtime not found at ${TIGRIS_RUNTIME_DIR}\n"
        "Set -DTIGRIS_RUNTIME_DIR=/path/to/tigris-runtime")
endif()

idf_component_register(
    SRCS "${TIGRIS_RUNTIME_DIR}/src/tigris_loader.c"
         "${TIGRIS_RUNTIME_DIR}/src/tigris_mem.c"
         "${TIGRIS_RUNTIME_DIR}/src/tigris_executor.c"
         "${TIGRIS_RUNTIME_DIR}/src/tigris_kernels.c"
         "${TIGRIS_RUNTIME_DIR}/src/tigris_kernels_s8.c"
         "${TIGRIS_RUNTIME_DIR}/src/tigris_kernels_esp_nn.c"
         "${TIGRIS_RUNTIME_DIR}/src/tigris_lz4.c"
    INCLUDE_DIRS "${TIGRIS_RUNTIME_DIR}/include"
)

# Enable ESP-NN backend when esp-nn component is available
idf_build_get_property(components BUILD_COMPONENTS)
if("esp-nn" IN_LIST components)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC TIGRIS_HAS_ESP_NN)
    idf_component_get_property(esp_nn_lib esp-nn COMPONENT_LIB)
    target_link_libraries(${COMPONENT_LIB} PUBLIC ${esp_nn_lib})
endif()
