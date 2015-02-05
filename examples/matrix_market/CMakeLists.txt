#cuda_include_directories(../)
#cuda_include_directories(../..)

SOURCE_GROUP("Spike Headers" FILES ${SPIKE_HEADERS})
SOURCE_GROUP("Spike CUDA Headers" FILES ${SPIKE_CUHEADERS})

cuda_add_executable(driver_mm driver_mm.cu ${SPIKE_HEADERS} ${SPIKE_CUHEADERS})
target_link_libraries(driver_mm cusparse)
