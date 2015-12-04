# Minimum version of CMake required. Don't touch.
cmake_minimum_required(VERSION 2.8)

# Set the name of your project
project(sap_examples)

# Include macros from the SBEL utils library
include(cmake/SBELUtils.cmake)

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


SET(SAP_HEADERS
	../../sap/bicgstab2.h
	../../sap/bicgstab.h
	../../sap/minres.h
	../../sap/common.h
	../../sap/exception.h
	../../sap/graph.h
	../../sap/monitor.h
	../../sap/precond.h
	../../sap/solver.h
	../../sap/spmv.h
	../../sap/strided_range.h
	../../sap/timer.h
	../../sap/segmented_matrix.h
)

SET(SAP_CUHEADERS
	../../sap/device/data_transfer.cuh
	../../sap/device/factor_band_const.cuh
	../../sap/device/factor_band_var.cuh
	../../sap/device/inner_product.cuh
	../../sap/device/matrix_multiply.cuh
	../../sap/device/shuffle.cuh
	../../sap/device/sweep_band_const.cuh
	../../sap/device/sweep_band_var.cuh
	../../sap/device/sweep_band_sparse.cuh
	../../sap/device/db.cuh
)

ADD_SUBDIRECTORY(matrix_market)
ADD_SUBDIRECTORY(views)
ADD_SUBDIRECTORY(restart)
ADD_SUBDIRECTORY(sequence)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(test_double)
ADD_SUBDIRECTORY(test_db)
ADD_SUBDIRECTORY(synthetic_banded)
ADD_SUBDIRECTORY(multi_gpu)
ADD_SUBDIRECTORY(dual_gpu_update)
ADD_SUBDIRECTORY(unit_test)
#ADD_SUBDIRECTORY(synthetic_sparse)
