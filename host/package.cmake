get_filename_component(library_name "${LIBRARY}" NAME)
file(SHA256 "${LIBRARY}" library_hash)
set(stage "${OUTPUT}/stage")
file(MAKE_DIRECTORY "${stage}")
configure_file("${LIBRARY}" "${stage}/${library_name}" COPYONLY)
configure_file("${HEADER}" "${stage}/tigris_host.h" COPYONLY)
configure_file("${LICENSE}" "${stage}/LICENSE" COPYONLY)
file(WRITE "${stage}/manifest.json"
    "{\n  \"version\": \"${VERSION}\",\n  \"abi\": 1,\n  \"platform\": \"${PLATFORM}\",\n  \"source_revision\": \"${REVISION}\",\n  \"library\": \"${library_name}\",\n  \"sha256\": \"${library_hash}\"\n}\n")
set(archive "tigris-host-${VERSION}-${PLATFORM}.tar.gz")
execute_process(COMMAND "${CMAKE_COMMAND}" -E tar czf "${OUTPUT}/${archive}"
    "${library_name}" tigris_host.h LICENSE manifest.json
    WORKING_DIRECTORY "${stage}" RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Host archive creation failed")
endif()
file(SHA256 "${OUTPUT}/${archive}" archive_hash)
file(WRITE "${OUTPUT}/${archive}.sha256" "${archive_hash}  ${archive}\n")
