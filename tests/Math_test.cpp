#include <cumulative_histogram/Math.h>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace CumulativeHistogram_NS {
namespace Detail_NS {
namespace {

TEST(CumulativeHistogramImpl, floorLog2) {
  for (std::size_t i = 1; i <= 128; ++i) {
    EXPECT_EQ(floorLog2(i), static_cast<std::size_t>(std::floor(std::log2(static_cast<double>(i)))));
  }
  // The standard guaraneets that std::size_t is at least 16 bits
  EXPECT_EQ(floorLog2(65535), 15);
  // Check the maximum value ((2^N) - 1).
  EXPECT_EQ(floorLog2(std::numeric_limits<std::size_t>::max()),
            std::numeric_limits<std::size_t>::digits - 1);
}

TEST(CumulativeHistogramImpl, ceilLog2) {
  for (std::size_t i = 1; i <= 128; ++i) {
    EXPECT_EQ(ceilLog2(i), static_cast<std::size_t>(std::ceil(std::log2(static_cast<double>(i)))));
  }
  // The standard guaraneets that std::size_t is at least 16 bits
  EXPECT_EQ(ceilLog2(65535), 16);
  // Check the maximum value ((2^N) - 1).
  EXPECT_EQ(ceilLog2(std::numeric_limits<std::size_t>::max()),
    std::numeric_limits<std::size_t>::digits);
}

TEST(CumulativeHistogramImpl, countElementsInLeftmostSubtree) {
  // Check all valid levels for N from [0; 1024]
  for (std::size_t num_elements = 0; num_elements <= 1024; ++num_elements) {
    std::size_t expected_num_elements_at_level = num_elements;
    std::size_t level = 0;
    while (true) {
      EXPECT_EQ(countElementsInLeftmostSubtree(num_elements, level), expected_num_elements_at_level);
      if (expected_num_elements_at_level < 2) {
        break;
      }
      const std::size_t num_elements_right = expected_num_elements_at_level / 2; // floor(N/2);
      const std::size_t num_elements_left = expected_num_elements_at_level - num_elements_right; // ceil(N/2)
      expected_num_elements_at_level = num_elements_left;
      ++level;
    }
  }
  // Check that overflow doesn't happen.
  constexpr std::size_t n_max = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(countElementsInLeftmostSubtree(n_max, 0), n_max);
  EXPECT_EQ(countElementsInLeftmostSubtree(n_max, 1), (n_max / 2) + (n_max % 2 != 0));
  EXPECT_EQ(countElementsInLeftmostSubtree(n_max, 2), (n_max / 4) + (n_max % 4 != 0));
}

TEST(CumulativeHistogramImpl, findDeepestNodeForElements) {
  // Slow (O(logN)) version for testing.
  const auto findDeepestNodeForElementsTest = [](std::size_t num_elements, std::size_t capacity) {
    std::size_t depth = 0;
    while (capacity > 1) {
      // Compute the capacity of the left subtree.
      const std::size_t capacity_left = (capacity / 2) + (capacity % 2 != 0);  // ceil(capacity/2)
      if (capacity_left < num_elements) {
        break;
      }
      ++depth;
      capacity = capacity_left;
    }
    return depth;
  };
  for (std::size_t capacity = 0; capacity <= 1024; ++capacity) {
    for (std::size_t num_elements = 0; num_elements <= capacity; ++num_elements) {
      EXPECT_EQ(findDeepestNodeForElements(num_elements, capacity),
                findDeepestNodeForElementsTest(num_elements, capacity));
    }
  }
  // Check that overflow doesn't happen.
  constexpr std::size_t n_max = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(findDeepestNodeForElements(n_max, n_max), 0);
  EXPECT_EQ(findDeepestNodeForElements(n_max / 2, n_max), 1);
  EXPECT_EQ(findDeepestNodeForElements(n_max / 4, n_max), 2);
}

}
}
}
