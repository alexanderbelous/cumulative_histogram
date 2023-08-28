#include <cumulative_histogram/CumulativeHistogram.h>

#include <array>
#include <numeric>
#include <sstream>
#include <gtest/gtest.h>

namespace CumulativeHistogram_NS {
namespace {

template<class T>
testing::AssertionResult CheckPrefixSums(const CumulativeHistogram<T>& histogram) {
  const std::span<const T> elements = histogram.elements();
  T expected {};
  for (std::size_t i = 0; i < elements.size(); ++i) {
    expected += elements[i];
    const T actual = histogram.prefixSum(i);
    if (actual != expected) {
      return testing::AssertionFailure() <<
        "Expected prefixSum(" << i << ") to return " << expected << "; got " << actual;
    }
  }
  return testing::AssertionSuccess();
}

template<class T>
testing::AssertionResult CheckLowerBound(const CumulativeHistogram<T>& histogram, const T& value) {
  // Find the lower bound safely in O(N).
  auto iter_expected = histogram.begin();
  T prefix_sum {};
  while (iter_expected != histogram.end()) {
    prefix_sum += *iter_expected;
    if (!(prefix_sum < value)) {
      break;
    }
    ++iter_expected;
  }
  const T prefix_sum_expected = (iter_expected == histogram.end()) ? T{} : prefix_sum;
  // Check that CumulativeHistogram::lowerBound() returns the same result.
  auto [iter_actual, prefix_sum_actual] = histogram.lowerBound(value);
  if ((iter_expected != iter_actual) || (prefix_sum_expected != prefix_sum_actual)) {
    return testing::AssertionFailure() << "Expected lowerBound(" << value << ") to return {"
      << iter_expected << ", " << prefix_sum_expected << "}; got {"
      << iter_actual << ", " << prefix_sum_actual << "}.";
  }
  return testing::AssertionSuccess();
}

template<class T>
testing::AssertionResult CheckUpperBound(const CumulativeHistogram<T>& histogram, const T& value) {
  // Find the upper bound safely in O(N).
  auto iter_expected = histogram.begin();
  T prefix_sum {};
  while (iter_expected != histogram.end()) {
    prefix_sum += *iter_expected;
    if (value < prefix_sum) {
      break;
    }
    ++iter_expected;
  }
  const T prefix_sum_expected = (iter_expected == histogram.end()) ? T{} : prefix_sum;
  // Check that CumulativeHistogram::upperBound() returns the same result.
  auto [iter_actual, prefix_sum_actual] = histogram.upperBound(value);
  if ((iter_expected != iter_actual) || (prefix_sum_expected != prefix_sum_actual)) {
    return testing::AssertionFailure() << "Expected upperBound(" << value << ") to return {"
      << iter_expected << ", " << prefix_sum_expected << "}; got {"
      << iter_actual << ", " << prefix_sum_actual << "}.";
  }
  return testing::AssertionSuccess();
}



TEST(CumulativeHistogram, DefaultConstructor) {
  CumulativeHistogram<int> histogram;
  EXPECT_EQ(histogram.size(), 0);
  EXPECT_TRUE(histogram.elements().empty());
}

TEST(CumulativeHistogram, NumElements) {
  for (std::size_t i = 0; i < 10; ++i) {
    CumulativeHistogram<int> histogram(i);
    EXPECT_EQ(histogram.size(), i);
  }
}

TEST(CumulativeHistogram, ZeroInitialization) {
  static constexpr std::size_t kNumElements = 5;
  CumulativeHistogram<int> histogram(kNumElements);
  for (std::size_t i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(histogram.element(i), 0);
  }
}

TEST(CumulativeHistogram, ConstructFromSizeAndValue) {
  constexpr std::size_t kNumElements = 5;
  constexpr int kValue = 1;
  CumulativeHistogram<int> histogram(kNumElements, kValue);
  EXPECT_EQ(histogram.size(), kNumElements);
  for (std::size_t i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(histogram.element(i), kValue);
  }
  EXPECT_EQ(histogram.totalSum(), kNumElements * kValue);
  EXPECT_TRUE(CheckPrefixSums(histogram));
}

TEST(CumulativeHistogram, ConstructFromVector) {
  const std::vector<int> elements = {1, 2, 3, 4, 5, 6, 7};
  CumulativeHistogram<int> histogram{ std::vector<int>{elements} };
  EXPECT_EQ(histogram.size(), elements.size());
  EXPECT_EQ(histogram.totalSum(), std::accumulate(elements.begin(), elements.end(), 0));
  EXPECT_TRUE(CheckPrefixSums(histogram));
}

TEST(CumulativeHistogram, ConstructFromMultiPassRange) {
  constexpr std::array<int, 7> elements = {1, 2, 3, 4, 5, 6, 7};
  CumulativeHistogram<int> histogram{elements.begin(), elements.end()};
  EXPECT_EQ(histogram.size(), elements.size());
  EXPECT_EQ(histogram.totalSum(), std::accumulate(elements.begin(), elements.end(), 0));
  EXPECT_TRUE(CheckPrefixSums(histogram));
}

TEST(CumulativeHistogram, ConstructFromSinglePassRange) {
  std::istringstream str("1 2 3 4 5");
  CumulativeHistogram<int> histogram {std::istream_iterator<int>(str), std::istream_iterator<int>()};
  EXPECT_EQ(histogram.size(), 5);
  EXPECT_EQ(histogram.element(0), 1);
  EXPECT_EQ(histogram.element(1), 2);
  EXPECT_EQ(histogram.element(2), 3);
  EXPECT_EQ(histogram.element(3), 4);
  EXPECT_EQ(histogram.element(4), 5);
  EXPECT_EQ(histogram.totalSum(), 15);
  EXPECT_TRUE(CheckPrefixSums(histogram));
}

TEST(CumulativeHistogram, Iterators) {
  constexpr std::array<unsigned int, 5> kElements = { 1, 2, 3, 4, 5 };
  CumulativeHistogram<unsigned int> histogram(kElements.begin(), kElements.end());
  EXPECT_NE(histogram.begin(), histogram.end());
  EXPECT_EQ(std::distance(histogram.begin(), histogram.end()), 5);
  auto it = histogram.begin();
  for (std::size_t i = 0; i < kElements.size(); ++i) {
    EXPECT_EQ(*it, kElements[i]);
    ++it;
  }
}

TEST(CumulativeHistogram, ReverseIterators) {
  constexpr std::array<unsigned int, 5> kElements = { 1, 2, 3, 4, 5 };
  CumulativeHistogram<unsigned int> histogram(kElements.begin(), kElements.end());
  EXPECT_NE(histogram.rbegin(), histogram.rend());
  EXPECT_EQ(std::distance(histogram.rbegin(), histogram.rend()), 5);
  auto it = histogram.rbegin();
  for (auto it_arr = kElements.rbegin(); it_arr != kElements.rend(); ++it_arr) {
    EXPECT_EQ(*it, *it_arr);
    ++it;
  }
}

TEST(CumulativeHistogram, Clear) {
  constexpr std::size_t kNumElements = 5;
  // Construct a histogram capable of storing 5 elements.
  CumulativeHistogram<unsigned int> histogram(kNumElements);
  EXPECT_EQ(histogram.capacity(), kNumElements);
  EXPECT_EQ(histogram.size(), kNumElements);
  // Remove all elements.
  histogram.clear();
  EXPECT_EQ(histogram.capacity(), kNumElements);
  EXPECT_EQ(histogram.size(), 0);
  // Push back 5 new elements {0, 1, 2, 3, 4}.
  for (std::size_t i = 0; i < kNumElements; ++i) {
    histogram.push_back();
    histogram.increment(i, i);
    EXPECT_EQ(histogram.capacity(), kNumElements);
    EXPECT_EQ(histogram.size(), i + 1);
    EXPECT_EQ(histogram.element(i), i);
  }
}

TEST(CumulativeHistogram, Reserve) {
  constexpr std::array<unsigned int, 5> kElements = { 1, 2, 3, 4, 5};
  // Construct a histogram capable of storing 5 elements.
  CumulativeHistogram<unsigned int> histogram(kElements.begin(), kElements.end());
  // Reserve memory for more elements than the histogram can currently store.
  const std::size_t capacity_old = histogram.capacity();
  const std::size_t capacity_new = capacity_old * 2;
  histogram.reserve(capacity_new);
  // The number of elements should remain the same.
  EXPECT_EQ(histogram.size(), kElements.size());
  // The new capacity should not be less than requested.
  EXPECT_GE(histogram.capacity(), capacity_new);
  // The elements should remain the same.
  for (std::size_t i = 0; i < kElements.size(); ++i) {
    EXPECT_EQ(histogram.element(i), kElements[i]);
  }
  // Total sum must remain the same.
  EXPECT_EQ(histogram.totalSum(), std::accumulate(kElements.begin(), kElements.end(), 0u));
  // Validate prefix sums.
  EXPECT_TRUE(CheckPrefixSums(histogram));
}

TEST(CumulativeHistogram, PushBackZeroInitialized) {
  CumulativeHistogram<unsigned int> histogram;
  histogram.push_back();
  EXPECT_EQ(histogram.size(), 1);
  EXPECT_EQ(histogram.element(0), 0);
  EXPECT_EQ(histogram.totalSum(), 0);
  EXPECT_TRUE(CheckPrefixSums(histogram));
  histogram.increment(0, 42);
  EXPECT_EQ(histogram.element(0), 42);
  EXPECT_EQ(histogram.totalSum(), 42);
  EXPECT_TRUE(CheckPrefixSums(histogram));
  histogram.push_back();
  EXPECT_EQ(histogram.size(), 2);
  EXPECT_EQ(histogram.element(0), 42);
  EXPECT_EQ(histogram.element(1), 0);
  EXPECT_EQ(histogram.totalSum(), 42);
  EXPECT_TRUE(CheckPrefixSums(histogram));
  histogram.increment(1, 5);
  EXPECT_EQ(histogram.element(0), 42);
  EXPECT_EQ(histogram.element(1), 5);
  EXPECT_EQ(histogram.totalSum(), 47);
  EXPECT_TRUE(CheckPrefixSums(histogram));
}

TEST(CumulativeHistogram, PopBack) {
  constexpr std::array<unsigned int, 10> kElements = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  CumulativeHistogram<unsigned int> histogram{ kElements.begin(), kElements.end() };
  const std::size_t initial_capacity = histogram.capacity();
  std::size_t num_elements = kElements.size();
  // Remove all elements one by one.
  do {
    --num_elements;
    histogram.pop_back();
    // The number of elements should decrease by 1.
    EXPECT_EQ(histogram.size(), num_elements);
    // Capacity should remain unchanged.
    EXPECT_EQ(histogram.capacity(), initial_capacity);
    // Elements [0; num_elements) must not change.
    for (std::size_t i = 0; i < num_elements; ++i) {
      EXPECT_EQ(histogram.element(i), kElements[i]);
    }
    // Check prefix sums for indices [0; num_elements)
    EXPECT_TRUE(CheckPrefixSums(histogram));
  } while (num_elements > 0);
  EXPECT_TRUE(histogram.empty());
  EXPECT_EQ(histogram.size(), 0);
}

TEST(CumulativeHistogram, ResizeSameSize) {
  constexpr std::array<unsigned int, 10> kElements = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  CumulativeHistogram<unsigned int> histogram{ kElements.begin(), kElements.end() };
  const std::span<const unsigned int> elements_old = histogram.elements();
  const std::size_t capacity_old = histogram.capacity();
  histogram.resize(kElements.size());
  // Size should remain the same.
  EXPECT_EQ(histogram.size(), kElements.size());
  // Capacity should remain the same.
  EXPECT_EQ(histogram.capacity(), capacity_old);
  // Check that the elements haven't been reallocated.
  EXPECT_EQ(histogram.elements().data(), elements_old.data());
  // Elements [0; 10) must remain the same.
  for (std::size_t i = 0; i < kElements.size(); ++i) {
    EXPECT_EQ(histogram.element(i), kElements[i]);
  }
  // Check prefix sums.
  EXPECT_TRUE(CheckPrefixSums(histogram));
}

TEST(CumulativeHistogram, ResizeToFewerElements) {
  constexpr std::array<unsigned int, 13> kElements = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
  CumulativeHistogram<unsigned int> histogram{ kElements.begin(), kElements.end() };
  const std::span<const unsigned int> elements_old = histogram.elements();
  const std::size_t capacity_old = histogram.capacity();
  constexpr std::size_t kNewSize = 5;
  histogram.resize(kNewSize);
  // Size should become kNewSize.
  EXPECT_EQ(histogram.size(), kNewSize);
  // Capacity should remain the same.
  EXPECT_EQ(histogram.capacity(), capacity_old);
  // Check that the elements haven't been reallocated.
  EXPECT_EQ(histogram.elements().data(), elements_old.data());
  // Elements [0; kNewSize) must remain the same.
  for (std::size_t i = 0; i < kNewSize; ++i) {
    EXPECT_EQ(histogram.element(i), kElements[i]);
  }
  // Validate the total sum.
  EXPECT_EQ(histogram.totalSum(), std::accumulate(kElements.begin(), kElements.begin() + kNewSize, 0u));
  // Check prefix sums.
  EXPECT_TRUE(CheckPrefixSums(histogram));
}

TEST(CumulativeHistogram, ResizeToMoreElementsWithinCapacity) {
  constexpr std::array<unsigned int, 13> kElements = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
  CumulativeHistogram<unsigned int> histogram{ kElements.begin(), kElements.end() };
  const std::span<const unsigned int> elements_old = histogram.elements();
  const std::size_t capacity_old = histogram.capacity();
  // Resize to 5 elements.
  const std::size_t kNewSize = 5;
  histogram.resize(5);
  // Resize back to 13 elements.
  histogram.resize(kElements.size());
  // Size should become kNewSize.
  EXPECT_EQ(histogram.size(), kElements.size());
  // Capacity should remain the same.
  EXPECT_EQ(histogram.capacity(), capacity_old);
  // Check that the elements haven't been reallocated.
  const std::span<const unsigned int> elements_new = histogram.elements();
  EXPECT_EQ(histogram.elements().data(), elements_old.data());
  // Elements [0; kNewSize) must remain the same.
  for (std::size_t i = 0; i < kNewSize; ++i) {
    EXPECT_EQ(histogram.element(i), kElements[i]);
  }
  // Elements [kNewSize; kElements.size()) must now be 0.
  for (std::size_t i = kNewSize; i < kElements.size(); ++i) {
    EXPECT_EQ(histogram.element(i), 0);
  }
  // And set x[i] to 1 so that the total sum changes.
  histogram.increment(kElements.size() - 1, 1);
  // Validate the total sum.
  const unsigned int total_sum = std::accumulate(elements_new.begin(), elements_new.end(), 0u);
  EXPECT_EQ(histogram.totalSum(), total_sum);
  // Check prefix sums.
  EXPECT_TRUE(CheckPrefixSums(histogram));
}

TEST(CumulativeHistogram, ResizeToMoreElementsOutsideCapacity) {
  constexpr std::array<unsigned int, 5> kElements = { 1, 2, 3, 4, 5 };
  CumulativeHistogram<unsigned int> histogram{ kElements.begin(), kElements.end() };
  const std::span<const unsigned int> elements_old = histogram.elements();
  // Resize to more elements than the histogram can currently store.
  const std::size_t kNewSize = histogram.capacity() * 2;
  histogram.resize(kNewSize);
  // Size should become kNewSize.
  EXPECT_EQ(histogram.size(), kNewSize);
  // Capacity should become greater or equal to kNewSize.
  EXPECT_GE(histogram.capacity(), kNewSize);
  // Not checking if the elements have been reallocated, because in theory this is not guaranteed
  // if the implementation uses realloc().
  // Elements [0; kElements.size()) must remain the same.
  for (std::size_t i = 0; i < kElements.size(); ++i) {
    EXPECT_EQ(histogram.element(i), kElements[i]);
  }
  // Elements [kElements.size(), kNewSize) must now be 0.
  for (std::size_t i = kElements.size(); i < kNewSize; ++i) {
    EXPECT_EQ(histogram.element(i), 0);
  }
  // Total sum must remain the same.
  const unsigned int total_sum = std::accumulate(kElements.begin(), kElements.end(), 0u);
  EXPECT_EQ(histogram.totalSum(), total_sum);
  // Check prefix sums.
  EXPECT_TRUE(CheckPrefixSums(histogram));
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
  EXPECT_EQ(histogram.prefixSum(0), 0);
  histogram.increment(0, 42);
  EXPECT_EQ(histogram.prefixSum(0), 42);
}

TEST(CumulativeHistogram, TwoElements) {
  CumulativeHistogram<int> histogram(2);
  EXPECT_EQ(histogram.prefixSum(1), 0);
  histogram.increment(0, 3);
  histogram.increment(1, 7);
  EXPECT_EQ(histogram.prefixSum(0), 3);
  EXPECT_EQ(histogram.prefixSum(1), 10);
}

TEST(CumulativeHistogram, ThreeElements) {
  CumulativeHistogram<int> histogram(3);
  histogram.increment(0, 3);
  histogram.increment(1, 7);
  EXPECT_EQ(histogram.prefixSum(0), 3);
  EXPECT_EQ(histogram.prefixSum(1), 10);
}

TEST(CumulativeHistogram, PrefixSum) {
  static constexpr std::size_t kNumElements = 10;
  CumulativeHistogram<int> histogram(kNumElements);
  EXPECT_EQ(histogram.prefixSum(3), 0);
  EXPECT_EQ(histogram.prefixSum(8), 0);
  histogram.increment(0, 3);
  EXPECT_EQ(histogram.prefixSum(3), 3);
  EXPECT_EQ(histogram.prefixSum(8), 3);
  histogram.increment(1, 7);
  EXPECT_EQ(histogram.prefixSum(3), 10);
  EXPECT_EQ(histogram.prefixSum(8), 10);
  histogram.increment(5, 1);
  EXPECT_EQ(histogram.prefixSum(3), 10);
  EXPECT_EQ(histogram.prefixSum(8), 11);
  histogram.increment(9, 10);
  EXPECT_EQ(histogram.prefixSum(3), 10);
  EXPECT_EQ(histogram.prefixSum(8), 11);
  histogram.increment(3, 5);
  EXPECT_EQ(histogram.prefixSum(3), 15);
  EXPECT_EQ(histogram.prefixSum(8), 16);
}

TEST(CumulativeHistogram, LowerBound) {
  constexpr std::array<unsigned int, 9> kElements = {1, 2, 3, 4, 5, 0, 0, 1, 2};
  const CumulativeHistogram<unsigned int> histogram {kElements.begin(), kElements.end()};
  for (unsigned int value = 0; value < 20; ++value) {
    EXPECT_TRUE(CheckLowerBound(histogram, value));
  }
}

TEST(CumulativeHistogram, LowerBoundZeros) {
  constexpr std::array<unsigned int, 9> kElements = { 1, 0, 0, 0, 0, 0, 0, 0, 0 };
  const CumulativeHistogram<unsigned int> histogram{ kElements.begin(), kElements.end() };
  EXPECT_TRUE(CheckLowerBound(histogram, 0u));
  EXPECT_TRUE(CheckLowerBound(histogram, 1u));
}

TEST(CumulativeHistogram, UpperBound) {
  constexpr std::array<unsigned int, 9> kElements = {1, 2, 3, 4, 5, 0, 0, 1, 2};
  const CumulativeHistogram<unsigned int> histogram {kElements.begin(), kElements.end()};
  for (unsigned int value = 0; value < 20; ++value) {
    EXPECT_TRUE(CheckUpperBound(histogram, value));
  }
}

TEST(CumulativeHistogram, UpperBoundZeros) {
  constexpr std::array<unsigned int, 9> kElements = { 1, 0, 0, 0, 0, 0, 0, 0, 0 };
  const CumulativeHistogram<unsigned int> histogram{ kElements.begin(), kElements.end() };
  EXPECT_TRUE(CheckUpperBound(histogram, 0u));
  EXPECT_TRUE(CheckUpperBound(histogram, 1u));
}

TEST(CumulativeHistogram, IsSubtree) {
  using Detail_NS::findLeftmostSubtreeWithExactCapacity;
  constexpr std::size_t kNotSubtree = static_cast<std::size_t>(-1);

  static_assert(findLeftmostSubtreeWithExactCapacity(0, 2) == kNotSubtree);
  static_assert(findLeftmostSubtreeWithExactCapacity(0, 3) == kNotSubtree);
  static_assert(findLeftmostSubtreeWithExactCapacity(1, 2) == kNotSubtree);
  static_assert(findLeftmostSubtreeWithExactCapacity(1, 3) == kNotSubtree);

  static_assert(findLeftmostSubtreeWithExactCapacity(2, 3) == 1);  // 3 == 2 + 1
  static_assert(findLeftmostSubtreeWithExactCapacity(2, 4) == 1);  // 4 == 2 + 2

  static_assert(findLeftmostSubtreeWithExactCapacity(6, 6) == 0);  // 6 == 6
  static_assert(findLeftmostSubtreeWithExactCapacity(6, 7) == kNotSubtree);

  static_assert(findLeftmostSubtreeWithExactCapacity(6, 11) == 1); // 11 == 6 + 5
  static_assert(findLeftmostSubtreeWithExactCapacity(6, 12) == 1); // 12 == 6 + 6
  static_assert(findLeftmostSubtreeWithExactCapacity(6, 10) == kNotSubtree);

  static_assert(findLeftmostSubtreeWithExactCapacity(6, 21) == 2); // 21 == 6 + 5 + 5 + 5
  static_assert(findLeftmostSubtreeWithExactCapacity(6, 22) == 2); // 22 == 6 + 5 + 6 + 5
  static_assert(findLeftmostSubtreeWithExactCapacity(6, 23) == 2); // 23 == 6 + 6 + 6 + 5
  static_assert(findLeftmostSubtreeWithExactCapacity(6, 24) == 2); // 24 == 6 + 6 + 6 + 6
  static_assert(findLeftmostSubtreeWithExactCapacity(6, 25) == kNotSubtree);
}

}
}