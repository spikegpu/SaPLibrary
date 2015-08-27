#cuda_include_directories(../)
#cuda_include_directories(../..)

SOURCE_GROUP("SaP Headers" FILES ${SAP_HEADERS})
SOURCE_GROUP("SaP CUDA Headers" FILES ${SAP_CUHEADERS})

cuda_add_executable(driver_banded driver_banded.cu ${SAP_HEADERS} ${SAP_CUHEADERS})
target_link_libraries(driver_banded cusparse)
