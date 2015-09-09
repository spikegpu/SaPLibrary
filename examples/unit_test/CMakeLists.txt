#cuda_include_directories(../)
#cuda_include_directories(../..)

SOURCE_GROUP("SaP Headers" FILES ${SAP_HEADERS})
SOURCE_GROUP("SaP CUDA Headers" FILES ${SAP_CUHEADERS})

########################################################################
# CMake build script for Google Test.
#
# To run the tests for Google Test itself on Linux, use 'make test' or
# ctest.  You can select which tests to run using 'ctest -R regex'.
# For more options, run 'ctest --help'.

# BUILD_SHARED_LIBS is a standard CMake variable, but we declare it here to
# make it prominent in the GUI.
option(BUILD_SHARED_LIBS "Build shared libraries (DLLs)." OFF)

# When other libraries are using a shared version of runtime libraries,
# Google Test also has to use one.
option(
  gtest_force_shared_crt
  "Use shared (DLL) run-time lib even when Google Test is built as static lib."
  OFF)

option(gtest_build_tests "Build all of gtest's own tests." OFF)

option(gtest_build_samples "Build gtest's sample programs." OFF)

option(gtest_disable_pthreads "Disable uses of pthreads in gtest." OFF)

option(gmock_build_tests "Build all of Google Mock's own tests." OFF)

# Defines pre_project_set_up_hermetic_build() and set_up_hermetic_build().
include(cmake/hermetic_build.cmake OPTIONAL)

if (COMMAND pre_project_set_up_hermetic_build)
  pre_project_set_up_hermetic_build()
endif()

if (COMMAND set_up_hermetic_build)
  set_up_hermetic_build()
endif()

# Define helper functions and macros used by Google Test.
include(../cmake/internal_utils.cmake)

config_compiler_and_linker()  # Defined in internal_utils.cmake.

find_path(gtest_SOURCE_DIR
    NAMES gtest.h
    DOC "gtest source directory")

find_path(gmock_SOURCE_DIR
    NAMES gmock.h
    DOC "gmock source directory")

if(gtest_SOURCE_DIR AND gmock_SOURCE_DIR)
    if(NOT (EXISTS "${gtest_SOURCE_DIR}/" AND EXISTS "${gtest_SOURCE_DIR}/include/"))
        message(WARNING "gtest headers not found. Please epecify or skip unit test.")
    elseif(NOT (EXISTS "${gmock_SOURCE_DIR}/" AND EXISTS "${gmock_SOURCE_DIR}/include/"))
        message(WARNING "gmock headers not found. Please epecify or skip unit test.")
    else()
        include_directories(
            ${gmock_SOURCE_DIR}
            ${gmock_SOURCE_DIR}/include
            ${gtest_SOURCE_DIR}/include
            ${gtest_SOURCE_DIR})

        # Where Google Test's .h files can be found.
        cuda_include_directories(
            ${gmock_SOURCE_DIR}
            ${gmock_SOURCE_DIR}/include
            ${gtest_SOURCE_DIR}/include
            ${gtest_SOURCE_DIR})

        # Where Google Test's libraries can be found.
        # link_directories(${gtest_BINARY_DIR}/src)

        ########################################################################
        #
        # Defines the gtest & gtest_main libraries.  User tests should link
        # with one of them.

        # Google Test libraries.  We build them using more strict warnings than what
        # are used for other targets, to ensure that gtest can be compiled by a user
        # aggressive about warnings.
        cxx_library(gtest "${cxx_strict}" ${gtest_SOURCE_DIR}/src/gtest-all.cc)
        cxx_library(gtest_main "${cxx_strict}" ${gtest_SOURCE_DIR}/src/gtest_main.cc)

        # Google Mock libraries.  We build them using more strict warnings than what
        # are used for other targets, to ensure that Google Mock can be compiled by
        # a user aggressive about warnings.
        cxx_library(gmock
            "${cxx_strict}"
            "${gmock_SOURCE_DIR}/src/gmock-all.cc")

        cxx_library(gmock_main
            "${cxx_strict}"
            "${gmock_SOURCE_DIR}/src/gmock_main.cc")

        target_link_libraries(gtest_main gtest)
        target_link_libraries(gmock gtest)
        target_link_libraries(gmock_main gmock)
        cuda_add_executable(driver_unit_test driver_unit_test.cu ${SAP_HEADERS} ${SAP_CUHEADERS})
        target_link_libraries(driver_unit_test cusparse gmock_main)
    endif()
else()
    message(WARNING "gtest or gmock headers not found. Please epecify or skip unit test.")
endif()
