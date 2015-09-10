#cuda_include_directories(../)
#cuda_include_directories(../..)

SOURCE_GROUP("SaP Headers" FILES ${SAP_HEADERS})
SOURCE_GROUP("SaP CUDA Headers" FILES ${SAP_CUHEADERS})

include_directories(thirdparty/googletest)
add_subdirectory(thirdparty/googletest)

cuda_add_executable(driver_unit_test driver_unit_test.cu ${SAP_HEADERS} ${SAP_CUHEADERS})
target_link_libraries(driver_unit_test cusparse googletest)
