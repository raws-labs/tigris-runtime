# Compares three symbol sets and fails unless they are equal: the arm_*,
# esp_nn_*, tigris_dispatch_kernel_s8 and tigris_accel_try_s8_ref symbols the
# accelerated runtime objects leave undefined (nm -u), the WRAPPED list, and
# the __wrap_* functions the wrapper objects define.
# Inputs: NM, WRAPPED (comma separated), OBJECTS and WRAPPERS (files listing
# object paths, one per line), STAMP (written on success).
cmake_minimum_required(VERSION 3.25)

function(read_lines file out)
    file(STRINGS "${file}" lines)
    list(FILTER lines EXCLUDE REGEX "^$")
    set(${out} "${lines}" PARENT_SCOPE)
endfunction()

function(nm_symbols objects flags pattern out)
    set(found "")
    foreach(object IN LISTS objects)
        execute_process(COMMAND "${NM}" ${flags} "${object}"
            RESULT_VARIABLE status OUTPUT_VARIABLE text ERROR_VARIABLE error)
        if(NOT status EQUAL 0)
            message(FATAL_ERROR "${NM} ${flags} ${object} failed: ${error}")
        endif()
        string(REGEX MATCHALL "${pattern}" names "${text}")
        list(APPEND found ${names})
    endforeach()
    list(TRANSFORM found REPLACE "^[ \t]*[A-Za-z][ \t]+" "")
    list(REMOVE_DUPLICATES found)
    list(SORT found)
    set(${out} "${found}" PARENT_SCOPE)
endfunction()

read_lines("${OBJECTS}" runtime_objects)
read_lines("${WRAPPERS}" wrapper_objects)
string(REPLACE "," ";" wrapped "${WRAPPED}")
list(SORT wrapped)

nm_symbols("${runtime_objects}" "-u" "[ \t]*U[ \t]+[A-Za-z0-9_]+" referenced)
list(FILTER referenced INCLUDE REGEX
    "^(arm_|esp_nn_)|^tigris_dispatch_kernel_s8$|^tigris_accel_try_s8_ref$")
nm_symbols("${wrapper_objects}" "--defined-only"
    "[0-9a-fA-F]+[ \t]+T[ \t]+__wrap_[A-Za-z0-9_]+" defined)
list(TRANSFORM defined REPLACE "^[0-9a-fA-F]+[ \t]+T[ \t]+__wrap_" "")
list(SORT defined)

set(failed FALSE)
foreach(symbol IN LISTS referenced)
    if(NOT symbol IN_LIST wrapped)
        message(SEND_ERROR "symbol ${symbol} is called by the runtime but not wrapped")
        set(failed TRUE)
    endif()
endforeach()
foreach(symbol IN LISTS wrapped)
    if(NOT symbol IN_LIST referenced)
        message(SEND_ERROR "wrapped symbol ${symbol} is not called by the runtime")
        set(failed TRUE)
    endif()
    if(NOT symbol IN_LIST defined)
        message(SEND_ERROR "wrapped symbol ${symbol} has no __wrap_${symbol}")
        set(failed TRUE)
    endif()
endforeach()
foreach(symbol IN LISTS defined)
    if(NOT symbol IN_LIST wrapped)
        message(SEND_ERROR "__wrap_${symbol} is defined but ${symbol} is not wrapped")
        set(failed TRUE)
    endif()
endforeach()
if(failed)
    message(FATAL_ERROR "call wrappers are out of date")
endif()
list(LENGTH wrapped count)
file(WRITE "${STAMP}" "${count} symbols wrapped\n")
