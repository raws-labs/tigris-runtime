idf_component_register(
    SRCS "main.c" "plan.S"
    INCLUDE_DIRS "."
    # The managed component pulled from the registry (namespace__name). It brings
    # in esp-nn and defines TIGRIS_HAS_ESP_NN, so the encoder convs are ESP-NN
    # accelerated with no extra wiring.
    REQUIRES raws-labs__tigris-runtime
    PRIV_REQUIRES esp_timer heap
    # The compiled plan (plan.S, 16-byte aligned) and its golden output are
    # embedded in the app, so the example runs with no partition flashing.
    EMBED_FILES "unet_ref.bin"
)
set_source_files_properties(plan.S PROPERTIES
    COMPILE_OPTIONS "-Wa,-I${CMAKE_CURRENT_SOURCE_DIR}"
    OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unet.tgrs")
