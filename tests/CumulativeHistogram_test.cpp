#include <cumulative_histogram/CumulativeHistogram.h>

#include <gtest/gtest.h>

namespace CumulativeHistogram_NS {
namespace {

TEST(CumulativeHistogram, DefaultConstructor) {
  CumulativeHistogram<int> histogram;
  EXPECT_EQ(histogram.numElements(), 0);
  EXPECT_TRUE(histogram.elements().empty());
}

TEST(CumulativeHistogram, NumElements) {
  for (std::size_t i = 0; i < 10; ++i) {
    CumulativeHistogram<int> histogram(i);
    EXPECT_EQ(histogram.numElements(), i);
  }
}

TEST(CumulativeHistogram, ZeroInitialization) {
  static constexpr std::size_t kNumElements = 5;
  CumulativeHistogram<int> histogram(kNumElements);
  for (std::size_t i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(histogram.element(i), 0);
  }
}

TEST(CumulativeHistogram, ConstructFromVector) {
  const std::vector<int> elements = {1, 2, 3, 4, 5, 6, 7};
  CumulativeHistogram<int> histogram{ std::vector<int>{elements} };
  EXPECT_EQ(histogram.numElements(), elements.size());
  EXPECT_EQ(histogram.totalSum(), std::accumulate(elements.begin(), elements.end(), 0));
  for (std::size_t i = 0; i < elements.size(); ++i) {
    const int partial_sum = std::accumulate(elements.begin(), elements.begin() + i + 1, 0);
    EXPECT_EQ(histogram.partialSum(i), partial_sum);
  }
}

TEST(CumulativeHistogram, ConstructFromRange) {
  const std::vector<int> elements = {1, 2, 3, 4, 5, 6, 7};
  CumulativeHistogram<int> histogram{elements.begin(), elements.end()};
  EXPECT_EQ(histogram.numElements(), elements.size());
  EXPECT_EQ(histogram.totalSum(), std::accumulate(elements.begin(), elements.end(), 0));
  for (std::size_t i = 0; i < elements.size(); ++i) {
    const int partial_sum = std::accumulate(elements.begin(), elements.begin() + i + 1, 0);
    EXPECT_EQ(histogram.partialSum(i), partial_sum);
  }
}

TEST(CumulativeHistogram, TotalSum) {
  static constexpr std::size_t kNumElements = 5;
  CumulativeHistogram<int> histogram(kNumElements);
  EXPECT_EQ(histogram.totalSum(), 0);
  histogram.increment(0, 2);
  EXPECT_EQ(histogram.totalSum(), 2);
  histogram.increment(1, 1);
  EXPECT_EQ(histogram.totalSum(), 3);
  histogram.increment(4, 7);
  EXPECT_EQ(histogram.totalSum(), 10);
}

TEST(CumulativeHistogram, Increment) {
  static constexpr std::size_t kNumElements = 5;
  static constexpr int kNewValues[kNumElements] = {1, 2, 3, 4, 5};
  CumulativeHistogram<int> histogram(kNumElements);
  for (std::size_t i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(histogram.element(i), 0);
    histogram.increment(i, kNewValues[i]);
    EXPECT_EQ(histogram.element(i), kNewValues[i]);
  }
  for (std::size_t i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(histogram.element(i), kNewValues[i]);
  }
}

TEST(CumulativeHistogram, OneElement) {
  CumulativeHistogram<int> histogram(1);
  EXPECT_EQ(histogram.partialSum(0), 0);
  histogram.increment(0, 42);
  EXPECT_EQ(histogram.partialSum(0), 42);
}

TEST(CumulativeHistogram, TwoElements) {
  CumulativeHistogram<int> histogram(2);
  EXPECT_EQ(histogram.partialSum(1), 0);
  histogram.increment(0, 3);
  histogram.increment(1, 7);
  EXPECT_EQ(histogram.partialSum(0), 3);
  EXPECT_EQ(histogram.partialSum(1), 10);
}

TEST(CumulativeHistogram, ThreeElements) {
  CumulativeHistogram<int> histogram(3);
  histogram.increment(0, 3);
  histogram.increment(1, 7);
  EXPECT_EQ(histogram.partialSum(0), 3);
  EXPECT_EQ(histogram.partialSum(1), 10);
}

TEST(CumulativeHistogram, PartialSum) {
  static constexpr std::size_t kNumElements = 10;
  CumulativeHistogram<int> histogram(kNumElements);
  EXPECT_EQ(histogram.partialSum(3), 0);
  EXPECT_EQ(histogram.partialSum(8), 0);
  histogram.increment(0, 3);
  EXPECT_EQ(histogram.partialSum(3), 3);
  EXPECT_EQ(histogram.partialSum(8), 3);
  histogram.increment(1, 7);
  EXPECT_EQ(histogram.partialSum(3), 10);
  EXPECT_EQ(histogram.partialSum(8), 10);
  histogram.increment(5, 1);
  EXPECT_EQ(histogram.partialSum(3), 10);
  EXPECT_EQ(histogram.partialSum(8), 11);
  histogram.increment(9, 10);
  EXPECT_EQ(histogram.partialSum(3), 10);
  EXPECT_EQ(histogram.partialSum(8), 11);
  histogram.increment(3, 5);
  EXPECT_EQ(histogram.partialSum(3), 15);
  EXPECT_EQ(histogram.partialSum(8), 16);
}

}
}