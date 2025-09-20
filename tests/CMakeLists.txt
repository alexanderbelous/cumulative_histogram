# Unit tests
add_executable(CumulativeHistogram_test
    "CompressedPath_test.cpp"
    "CumulativeHistogram_test.cpp"
    "Math_test.cpp"
)
target_link_libraries(CumulativeHistogram_test PRIVATE cumulative_histogram::cumulative_histogram)
target_link_libraries(CumulativeHistogram_test PRIVATE gtest_main)
# Set warning level
if(MSVC)
  target_compile_options(CumulativeHistogram_test PRIVATE /W4 /permissive-)
else()
  target_compile_options(CumulativeHistogram_test PRIVATE -Wall -Wextra -Wpedantic)
endif()


# Benchmark
add_executable(CumulativeHistogram_benchmark
    "CumulativeHistogram_benchmark.cpp"
)
target_link_libraries(CumulativeHistogram_benchmark PRIVATE cumulative_histogram::cumulative_histogram)
target_link_libraries(CumulativeHistogram_benchmark PRIVATE benchmark::benchmark)
