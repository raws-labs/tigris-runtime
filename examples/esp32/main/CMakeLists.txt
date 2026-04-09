idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES tigris_runtime
    PRIV_REQUIRES esp_partition esp_timer heap
)

# Enable with: idf.py build -DTIGRIS_PROFILING=1
if(TIGRIS_PROFILING)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE TIGRIS_PROFILING)
endif()
