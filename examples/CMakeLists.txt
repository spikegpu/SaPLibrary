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

ADD_SUBDIRECTORY(matrix_market)
#ADD_SUBDIRECTORY(synthetic_banded)
#ADD_SUBDIRECTORY(synthetic_sparse)
