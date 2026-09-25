/* The plan is embedded on a 16-byte boundary: ESP-NN reads filters in place
 * from it, and its SIMD kernels need them 16-byte aligned. EMBED_FILES does
 * not align. */
    .section .rodata.tigris_plan, "a"
    .balign 16
    .global unet_tgrs_start
unet_tgrs_start:
    .incbin "unet.tgrs"
    .global unet_tgrs_end
unet_tgrs_end:
