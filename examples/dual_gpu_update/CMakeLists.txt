#cuda_include_directories(../)
#cuda_include_directories(../..)

SOURCE_GROUP("Spike Headers" FILES ${SPIKE_HEADERS})
SOURCE_GROUP("Spike CUDA Headers" FILES ${SPIKE_CUHEADERS})

cuda_add_executable(driver_dual_gpu_update driver_dual_gpu_update.cu ${SPIKE_HEADERS} ${SPIKE_CUHEADERS})
