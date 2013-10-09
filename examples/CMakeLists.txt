# Minimum version of CMake required. Don't touch.
cmake_minimum_required(VERSION 2.8)

# Set the name of your project
project(spike_examples)

# Include macros from the SBEL utils library
include(SBELUtils.cmake)

##########################################################
# If using Makefiles, be sure to set CMAKE_BUILD_TYPE to # 
# select different build configurations.                 #
# Possible values:                                       #
# - RELEASE - fully optimized for speed                  #
# - DEBUG - unoptimized, with debugging symbols          #
# - RELWITHDEBUGINFO - optimized, with debugging symbols #
# - MINSIZEREL - optimized for size                      #
# - [not set] - unoptimized                              #
##########################################################


#enable_mpi_support()
#enable_openmp_support()
enable_cuda_support()

include_directories(.)
include_directories(..)

cuda_include_directories(/usr/local/cuda/cusp)


SET(SPIKE_HEADERS
	../../spike/bicgstab2.h
	../../spike/common.h
	../../spike/components.h
	../../spike/graph.h
	../../spike/monitor.h
	../../spike/precond.h
	../../spike/solver.h
	../../spike/spmv.h
	../../spike/timer.h
)

SET(SPIKE_CUHEADERS
	../../spike/device/data_transfer.cuh
	../../spike/device/factor_band_const.cuh
	../../spike/device/factor_band_var.cuh
	../../spike/device/inner_product.cuh
	../../spike/device/shuffle.cuh
	../../spike/device/sweep_band_const.cuh
	../../spike/device/sweep_band_var.cuh
)

ADD_SUBDIRECTORY(matrix_market)
ADD_SUBDIRECTORY(views)
ADD_SUBDIRECTORY(sequence)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(synthetic_banded)
#ADD_SUBDIRECTORY(synthetic_sparse)
