cmake_minimum_required(VERSION 4.0)

# Requires C++20.
set(CMAKE_CXX_STANDARD 20 CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(
  cumulative_histogram
  VERSION 1.0.0
  LANGUAGES CXX
)

add_library(cumulative_histogram INTERFACE)
add_library(cumulative_histogram::cumulative_histogram ALIAS cumulative_histogram)

option(CUMULATIVE_HISTOGRAM_TEST "Build tests for cumulative_histogram" OFF)

# Setup include directory
add_subdirectory(include)

if (CUMULATIVE_HISTOGRAM_TEST)
  # Use static (MT/MTd) Windows runtime.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

  option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
  add_subdirectory(thirdparty/benchmark EXCLUDE_FROM_ALL)
  add_subdirectory(thirdparty/googletest EXCLUDE_FROM_ALL)
  add_subdirectory(tests)
endif()
