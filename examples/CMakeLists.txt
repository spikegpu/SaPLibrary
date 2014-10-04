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
enable_openmp_support()
enable_cuda_support()

# CUSP dependency. Note that we always include CUSP headers using 
# include <cusp/header.h>.
find_path(CUSP_INCLUDE_DIR 
          NAMES cusp/array1d.h 
          HINTS ${CUDA_TOOLKIT_INCLUDE} 
          DOC "Top-level CUSP directory")
if(CUSP_INCLUDE_DIR)
   message(STATUS "CUSP headers found in ${CUSP_INCLUDE_DIR}/cusp")
else()
   message(FATAL_ERROR "CUSP headers not found. Please set CUSP_INCLUDE_DIR.")
endif()
cuda_include_directories(${CUSP_INCLUDE_DIR})


include_directories(.)
include_directories(..)


SET(SPIKE_HEADERS
	../../spike/bicgstab2.h
	../../spike/bicgstab.h
	../../spike/minres.h
	../../spike/common.h
	../../spike/exception.h
	../../spike/graph.h
	../../spike/monitor.h
	../../spike/precond.h
	../../spike/solver.h
	../../spike/spmv.h
	../../spike/strided_range.h
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
	../../spike/device/sweep_band_sparse.cuh
	../../spike/device/mc64.cuh
)

ADD_SUBDIRECTORY(matrix_market)
ADD_SUBDIRECTORY(views)
ADD_SUBDIRECTORY(sequence)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(test_double)
ADD_SUBDIRECTORY(test_mc64)
ADD_SUBDIRECTORY(synthetic_banded)
#ADD_SUBDIRECTORY(synthetic_sparse)
