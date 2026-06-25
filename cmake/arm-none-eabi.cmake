# CMake toolchain file for bare-metal Arm Cortex-M cross-compilation with the
# GNU Arm Embedded toolchain (arm-none-eabi-gcc).
#
# Per-core flags (-mcpu / -mfpu / -mfloat-abi) are intentionally NOT set here.
# A single toolchain file is shared across the M7 / M4F / M33 targets, so the
# firmware project supplies those per board (see boards/<board>/board.cmake).
#
# Usage:
#   cmake -B build \
#     -DCMAKE_TOOLCHAIN_FILE=<runtime>/cmake/arm-none-eabi.cmake \
#     -DTIGRIS_BOARD=nucleo_h753zi
#
# Override the toolchain location with -DARM_TOOLCHAIN_PREFIX=/opt/.../arm-none-eabi-

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

if(NOT ARM_TOOLCHAIN_PREFIX)
    set(ARM_TOOLCHAIN_PREFIX arm-none-eabi-)
endif()

find_program(ARM_CC  ${ARM_TOOLCHAIN_PREFIX}gcc)
find_program(ARM_CXX ${ARM_TOOLCHAIN_PREFIX}g++)
find_program(ARM_OBJCOPY ${ARM_TOOLCHAIN_PREFIX}objcopy)
find_program(ARM_SIZE    ${ARM_TOOLCHAIN_PREFIX}size)

if(NOT ARM_CC)
    message(FATAL_ERROR
        "Could not find ${ARM_TOOLCHAIN_PREFIX}gcc on PATH.\n"
        "Install it (Ubuntu: sudo apt install gcc-arm-none-eabi) or pass\n"
        "-DARM_TOOLCHAIN_PREFIX=/path/to/arm-none-eabi-")
endif()

set(CMAKE_C_COMPILER   ${ARM_CC})
set(CMAKE_ASM_COMPILER ${ARM_CC})
set(CMAKE_CXX_COMPILER ${ARM_CXX})
set(CMAKE_OBJCOPY ${ARM_OBJCOPY} CACHE FILEPATH "arm objcopy")
set(CMAKE_SIZE    ${ARM_SIZE}    CACHE FILEPATH "arm size")

# The default compiler check links a full executable, which fails on bare metal
# without a linker script. Probe with a static library instead.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Look for headers/libraries only in the cross sysroot, but find programs (the
# compiler itself) on the host PATH.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
