project(gtest)
cmake_minimum_required(VERSION 2.6.2)

include_directories(
  .
  ./gtest
  ./gmock)
  

add_library(googletest STATIC gmock-gtest-all.cc)
