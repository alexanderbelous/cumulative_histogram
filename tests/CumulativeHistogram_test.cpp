#include <cumulative_histogram/CumulativeHistogram.h>

#include <array>
#include <complex>
#include <ostream>
#include <numeric>
#include <sstream>
#include <gtest/gtest.h>

namespace CumulativeHistogram_NS {

// Use bucket size = 2 for all types in these tests.
template<class T>
class BucketSize<T, void> : public std::integral_constant<std::size_t, 2> {};
// Sanity checks.
static_assert(BucketSize<int>::value == 2, "BucketSize<int> must be equal to 2 for unit tests.");
static_assert(BucketSize<unsigned int>::value == 2, "BucketSize<unsigned int > must be equal to 2 for unit tests.");

namespace {

template<class T, class SumOperation>
testing::AssertionResult CheckPrefixSums(const CumulativeHistogram<T, SumOperation>& histogram) {
  const SumOperation sum_op = histogram.sumOperation();
  const std::span<const T> elements = histogram.elements();
  T expected {};
  for (std::size_t i = 0; i < elements.size(); ++i) {
    expected = sum_op(std::move(expected), elements[i]);
    const T actual = histogram.prefixSum(i);
    if (actual != expected) {
      return testing::AssertionFailure() <<
        "Expected prefixSum(" << i << ") to return " << expected << "; got " << actual;
    }
  }
  const T total_sum_actual = histogram.totalSum();
  if (total_sum_actual != expected) {
    return testing::AssertionFailure() <<
      "Expected totalSum() to return " << expected << "; got " << total_sum_actual;
  }
  return testing::AssertionSuccess();
}

template<class T, class SumOperation>
testing::AssertionResult CheckLowerBound(const CumulativeHistogram<T, SumOperation>& histogram, const T& value) {
  const SumOperation sum_op = histogram.sumOperation();
  // Find the lower bound safely in O(N).
  auto iter_expected = histogram.begin();
  T prefix_sum {};
  while (iter_expected != histogram.end()) {
    prefix_sum = sum_op(std::move(prefix_sum), *iter_expected);
    if (!(prefix_sum < value)) {
      break;
    }
    ++iter_expected;
  }
  const T prefix_sum_expected = (iter_expected == histogram.end()) ? T{} : prefix_sum;
  // Check that CumulativeHistogram::lowerBound() returns the same result.
  auto [iter_actual, prefix_sum_actual] = histogram.lowerBound(value);
  if ((iter_expected != iter_actual) || (prefix_sum_expected != prefix_sum_actual)) {
    const std::ptrdiff_t index_expected = std::distance(histogram.begin(), iter_expected);
    const std::ptrdiff_t index_actual = std::distance(histogram.begin(), iter_actual);
    return testing::AssertionFailure() << "Expected lowerBound(" << value << ") to return {"
      << "begin() + " << index_expected << ", " << prefix_sum_expected << "}; got {"
      << "begin() + " << index_actual << ", " << prefix_sum_actual << "}.";
  }
  return testing::AssertionSuccess();
}

template<class T, class SumOperation>
testing::AssertionResult CheckUpperBound(const CumulativeHistogram<T, SumOperation>& histogram, const T& value) {
  const SumOperation sum_op = histogram.sumOperation();
  // Find the upper bound safely in O(N).
  auto iter_expected = histogram.begin();
  T prefix_sum {};
  while (iter_expected != histogram.end()) {
    prefix_sum = sum_op(std::move(prefix_sum), *iter_expected);
    if (value < prefix_sum) {
      break;
    }
    ++iter_expected;
  }
  const T prefix_sum_expected = (iter_expected == histogram.end()) ? T{} : prefix_sum;
  // Check that CumulativeHistogram::upperBound() returns the same result.
  auto [iter_actual, prefix_sum_actual] = histogram.upperBound(value);
  if ((iter_expected != iter_actual) || (prefix_sum_expected != prefix_sum_actual)) {
    const std::ptrdiff_t index_expected = std::distance(histogram.begin(), iter_expected);
    const std::ptrdiff_t index_actual = std::distance(histogram.begin(), iter_actual);
    return testing::AssertionFailure() << "Expected upperBound(" << value << ") to return {"
      << "begin() + " << index_expected << ", " << prefix_sum_expected << "}; got {"
      << "begin() + " << index_actual << ", " << prefix_sum_actual << "}.";
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

TEST(CumulativeHistogram, CopyAssignmentWithinCapacity)
{
  constexpr std::size_t capacity = 128;
  constexpr std::size_t num_elements_new = 47;
  static_assert(num_elements_new <= capacity);

  CumulativeHistogram<int> histogram1;
  histogram1.reserve(capacity);
  const int* const data_was = histogram1.elements().data();
  const CumulativeHistogram<int> histogram2(num_elements_new, 1);
  histogram1 = histogram2;
  EXPECT_EQ(histogram1.elements(), histogram2.elements());
  EXPECT_EQ(histogram1.capacity(), capacity);
  EXPECT_EQ(histogram1.elements().data(), data_was);
  EXPECT_TRUE(CheckPrefixSums(histogram1));
}

TEST(CumulativeHistogram, CopyAssignmentOutsideCapacity)
{
  constexpr std::size_t capacity = 128;
  constexpr std::size_t num_elements_new = 147;
  static_assert(num_elements_new > capacity);

  CumulativeHistogram<int> histogram1;
  histogram1.reserve(capacity);
  const CumulativeHistogram<int> histogram2(num_elements_new, 1);
  histogram1 = histogram2;
  EXPECT_EQ(histogram1.elements(), histogram2.elements());
  EXPECT_GE(histogram1.capacity(), num_elements_new);
  EXPECT_TRUE(CheckPrefixSums(histogram1));
}

TEST(CumulativeHistogram, CopyAssignmentSelf)
{
  constexpr std::size_t capacity = 128;
  constexpr std::size_t num_elements = 47;
  static_assert(num_elements <= capacity);

  CumulativeHistogram<int> histogram(num_elements, 1);
  histogram.reserve(capacity);
  const int* const data_was = histogram.elements().data();
  histogram = histogram;
  EXPECT_EQ(histogram.capacity(), capacity);
  EXPECT_EQ(histogram.size(), num_elements);
  EXPECT_EQ(histogram.elements().data(), data_was);
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

TEST(CumulativeHistogram, ElementsMove)
{
  constexpr std::size_t num_elements = 10;
  CumulativeHistogram<unsigned int> histogram(num_elements, 1);
  const unsigned int* const data_was = histogram.elements().data();
  EXPECT_EQ(histogram.size(), num_elements);
  // Move the elements out of the histogram.
  std::vector<unsigned int> elements = std::move(histogram).elements();
  EXPECT_EQ(elements.data(), data_was);
  EXPECT_EQ(elements.size(), num_elements);
  // The histogram must be empty now.
  EXPECT_TRUE(histogram.empty());
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
    histogram.pushBack(static_cast<unsigned int>(i));
    EXPECT_EQ(histogram.capacity(), kNumElements);
    EXPECT_EQ(histogram.size(), i + 1);
    EXPECT_EQ(histogram.element(i), i);
  }
}

TEST(CumulativeHistogram, SetZeroAtFullCapacity) {
  constexpr std::size_t kNumElements = 32;
  CumulativeHistogram<int> histogram(kNumElements, 1);
  histogram.setZero();
  for (std::size_t i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(histogram.element(i), 0);
    EXPECT_EQ(histogram.prefixSum(i), 0);
  }
}

TEST(CumulativeHistogram, SetZeroAtMoreThanHalfCapacity) {
  constexpr std::size_t kCapacity = 32;
  constexpr std::size_t kNumElements = kCapacity / 2 + 1;
  CumulativeHistogram<int> histogram(kNumElements, 1);
  histogram.reserve(kCapacity);
  histogram.setZero();
  for (std::size_t i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(histogram.element(i), 0);
  }
  CheckPrefixSums(histogram);
}

TEST(CumulativeHistogram, SetZeroAtLessThanHalfCapacity) {
  constexpr std::size_t kCapacity = 32;
  constexpr std::size_t kNumElements = kCapacity / 4 + 1;
  CumulativeHistogram<int> histogram(kNumElements, 1);
  histogram.reserve(kCapacity);
  histogram.setZero();
  for (std::size_t i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(histogram.element(i), 0);
  }
  CheckPrefixSums(histogram);
}

TEST(CumulativeHistogram, FillAtFullCapacity) {
  constexpr std::size_t kNumElements = 32;
  constexpr int kValueToFillWith = 2;
  CumulativeHistogram<int> histogram(kNumElements, 1);
  histogram.fill(kValueToFillWith);
  for (std::size_t i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(histogram.element(i), kValueToFillWith);
  }
  CheckPrefixSums(histogram);
}

TEST(CumulativeHistogram, FillAtMoreThanHalfCapacity) {
  constexpr std::size_t kCapacity = 32;
  constexpr std::size_t kNumElements = kCapacity / 2 + 1;
  constexpr int kValueToFillWith = 2;
  CumulativeHistogram<int> histogram(kNumElements, 1);
  histogram.reserve(kCapacity);
  histogram.fill(kValueToFillWith);
  for (std::size_t i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(histogram.element(i), kValueToFillWith);
  }
  CheckPrefixSums(histogram);
}

TEST(CumulativeHistogram, FillAtLessThanHalfCapacity) {
  constexpr std::size_t kCapacity = 32;
  constexpr std::size_t kNumElements = kCapacity / 4 + 1;
  constexpr int kValueToFillWith = 2;
  CumulativeHistogram<int> histogram(kNumElements, 1);
  histogram.reserve(kCapacity);
  histogram.fill(kValueToFillWith);
  for (std::size_t i = 0; i < kNumElements; ++i) {
    EXPECT_EQ(histogram.element(i), kValueToFillWith);
  }
  CheckPrefixSums(histogram);
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
  histogram.pushBack(0);
  EXPECT_EQ(histogram.size(), 1);
  EXPECT_EQ(histogram.element(0), 0);
  EXPECT_EQ(histogram.totalSum(), 0);
  EXPECT_TRUE(CheckPrefixSums(histogram));
  histogram.increment(0, 42);
  EXPECT_EQ(histogram.element(0), 42);
  EXPECT_EQ(histogram.totalSum(), 42);
  EXPECT_TRUE(CheckPrefixSums(histogram));
  histogram.pushBack(0);
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
  histogram.pushBack(0);
  EXPECT_EQ(histogram.size(), 3);
  EXPECT_EQ(histogram.element(0), 42);
  EXPECT_EQ(histogram.element(1), 5);
  EXPECT_EQ(histogram.element(2), 0);
  EXPECT_EQ(histogram.totalSum(), 47);
  EXPECT_TRUE(CheckPrefixSums(histogram));
  histogram.increment(2, 3);
  EXPECT_EQ(histogram.element(0), 42);
  EXPECT_EQ(histogram.element(1), 5);
  EXPECT_EQ(histogram.element(2), 3);
  EXPECT_EQ(histogram.totalSum(), 50);
  EXPECT_TRUE(CheckPrefixSums(histogram));
}

TEST(CumulativeHistogram, PushBackNoReallocation) {
  constexpr std::size_t kNumElementsMin = 1;
  constexpr std::size_t kNumElementsMax = 128;
  for (std::size_t capacity = kNumElementsMin; capacity < kNumElementsMax; ++capacity) {
    CumulativeHistogram<unsigned int> histogram;
    histogram.reserve(capacity);
    for (std::size_t num_elements = 0; num_elements < capacity;) {
      // Assign histogram[i] = i+1;
      const unsigned int value = static_cast<unsigned int>(num_elements + 1);
      histogram.pushBack(value);
      ++num_elements;
      // Check the number of elements.
      EXPECT_EQ(histogram.size(), num_elements);
      // Check that the capacity hasn't changed.
      EXPECT_EQ(histogram.capacity(), capacity);
      // Check the elements.
      for (std::size_t i = 0; i < num_elements; ++i) {
        EXPECT_EQ(histogram.element(i), static_cast<unsigned int>(i + 1));
      }
      // Check the total sum (1 + 2 + 3 + ... + N == N * (N+1) / 2).
      EXPECT_EQ(histogram.totalSum(), num_elements * (num_elements + 1) / 2);
      // Check the prefix sums.
      EXPECT_TRUE(CheckPrefixSums(histogram));
    }
  }
}

TEST(CumulativeHistogram, PushBackNonZero) {
  constexpr std::array<unsigned int, 17> kElements =
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 };
  CumulativeHistogram<unsigned int> histogram;
  for (std::size_t i = 0; i < kElements.size(); ++i) {
    histogram.pushBack(kElements[i]);
    const std::size_t new_size = i + 1;
    EXPECT_EQ(histogram.size(), new_size);
    // Check elements.
    for (std::size_t j = 0; j < new_size; ++j) {
      EXPECT_EQ(histogram.element(j), kElements[j]);
    }
    EXPECT_EQ(histogram.totalSum(), std::accumulate(kElements.begin(), kElements.begin() + new_size, 0u));
    EXPECT_TRUE(CheckPrefixSums(histogram));
  }
}

TEST(CumulativeHistogram, PopBack) {
  constexpr std::array<unsigned int, 10> kElements = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  CumulativeHistogram<unsigned int> histogram{ kElements.begin(), kElements.end() };
  const std::size_t initial_capacity = histogram.capacity();
  std::size_t num_elements = kElements.size();
  // Remove all elements one by one.
  do {
    --num_elements;
    histogram.popBack();
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

TEST(CumulativeHistogram, Swap) {
  constexpr std::size_t kNumElements1 = 10;
  constexpr std::size_t kNumElements2 = 20;
  CumulativeHistogram<unsigned int> histogram1(kNumElements1, 24601);
  CumulativeHistogram<unsigned int> histogram2(kNumElements2, 42);
  const unsigned int* const data1 = histogram1.elements().data();
  const unsigned int* const data2 = histogram2.elements().data();
  histogram1.swap(histogram2);
  EXPECT_EQ(histogram1.size(), kNumElements2);
  EXPECT_EQ(histogram2.size(), kNumElements1);
  EXPECT_EQ(histogram1.elements().data(), data2);
  EXPECT_EQ(histogram2.elements().data(), data1);
  EXPECT_TRUE(CheckPrefixSums(histogram1));
  EXPECT_TRUE(CheckPrefixSums(histogram2));
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
  using ElementType = unsigned int;
  constexpr std::size_t kCapacityMax = 40;
  constexpr ElementType kInitialValue = 1;
  constexpr ElementType kIncrementValue = 7;
  constexpr ElementType kNewValue = kInitialValue + kIncrementValue;
  for (std::size_t capacity = 0; capacity < kCapacityMax; ++capacity)
  {
    for (std::size_t num_elements = 0; num_elements <= capacity; ++num_elements)
    {
      CumulativeHistogram<ElementType> histogram(num_elements, kInitialValue);
      histogram.reserve(capacity);
      for (std::size_t i = 0; i < num_elements; ++i)
      {
        histogram.increment(i, kIncrementValue);
        // i-th element and all elements before it must be equal to kNewValue.
        for (std::size_t j = 0; j <= i; ++j)
        {
          EXPECT_EQ(histogram.element(j), kNewValue);
        }
        // All elements after i must still be equal to kInitialValue.
        for (std::size_t j = i + 1; j < num_elements; ++j)
        {
          EXPECT_EQ(histogram.element(j), kInitialValue);
        }
        // Check all prefix sums.
        EXPECT_TRUE(CheckPrefixSums(histogram));
      }
    }
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
  constexpr std::size_t kCapacityMin = 9;
  constexpr std::size_t kCapacityMax = 20;
  for (std::size_t capacity = kCapacityMin; capacity <= kCapacityMax; ++capacity) {
    CumulativeHistogram<unsigned int> histogram{ kElements.begin(), kElements.end() };
    histogram.reserve(capacity);
    for (unsigned int value = 0; value < 20; ++value) {
      EXPECT_TRUE(CheckLowerBound(histogram, value));
    }
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

  static_assert(findLeftmostSubtreeWithExactCapacity(0, 0) == 0);
  static_assert(findLeftmostSubtreeWithExactCapacity(0, 1) == kNotSubtree);
  static_assert(findLeftmostSubtreeWithExactCapacity(0, 2) == kNotSubtree);
  static_assert(findLeftmostSubtreeWithExactCapacity(0, 3) == kNotSubtree);
  static_assert(findLeftmostSubtreeWithExactCapacity(1, 2) == 1);
  static_assert(findLeftmostSubtreeWithExactCapacity(1, 3) == 2);

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

TEST(CumulativeHistogram, ComputeNewCapacityForFullTree) {
  // Safe, but slow version: find the smallest number M, such that
  //   M >= 2*num_elements
  //   and (ceil(M/bucket_size) = 2*ceil(M/bucket_size)
  //        or ceil(M/bucket_size) = 2*ceil(M/bucket_size)-1).
  const auto computeNewCapacityNaive = [](std::size_t num_elements, std::size_t bucket_size) -> std::size_t {
    // Edge case.
    if (num_elements == 0) {
      return 2;
    }
    const std::size_t num_buckets = Detail_NS::countBuckets(num_elements, bucket_size);
    const std::size_t m_min = 2 * num_elements;
    const std::size_t m_max = 2 * num_buckets * bucket_size;
    for (std::size_t m = m_min; m < m_max; ++m) {
      const std::size_t num_buckets_new = Detail_NS::countBuckets(m, bucket_size);
      if (num_buckets_new == 2 * num_buckets || num_buckets_new == 2 * num_buckets - 1) {
        return m;
      }
    }
    return m_max;
  };
  for (std::size_t bucket_size = 2; bucket_size <= 16; ++bucket_size) {
    constexpr std::size_t kMaxElements = 1024;
    for (std::size_t num_elements = 0; num_elements < kMaxElements; ++num_elements) {
      EXPECT_EQ(Detail_NS::computeNewCapacityForFullTree(num_elements),
                computeNewCapacityNaive(num_elements, bucket_size));
    }
  }
}

TEST(CumulativeHistogram, Complex) {
  constexpr std::array<std::complex<float>, 10> kElements = {{
    {1.0, 0.0},
    {0.0, 1.0},
    {1.5, -2.5},
    {-2.5, 3.25},
    {-4.0, 3.75},
    {0.25, -1.25},
    {-1.75, 4.5},
    {20.0, -7.0},
    {0.0, 0.0},
    {-0.25, 0.25}
  }};
  CumulativeHistogram<std::complex<float>> histogram;
  for (std::size_t i = 0; i < kElements.size(); ++i) {
    histogram.pushBack(kElements[i]);
    const std::size_t new_size = i + 1;
    EXPECT_EQ(histogram.size(), new_size);
    // Check elements.
    for (std::size_t j = 0; j < new_size; ++j) {
      EXPECT_EQ(histogram.element(j), kElements[j]);
    }
    EXPECT_EQ(histogram.totalSum(), std::accumulate(kElements.begin(), kElements.begin() + new_size, std::complex<float>(0.0)));
    EXPECT_TRUE(CheckPrefixSums(histogram));
    // Compilation error: std::complex<float> doesnt't have operator<.
    // auto [iter, prefix_sum] = histogram.lowerBound(std::complex<float>(0.25, 0.25));
  }
}

// User-defined type that satisfies the Semiregular concept.
struct CustomType
{
  unsigned int value;
};

// Inequality comparison operator for CustomType.
// This is not needed for the CumulativeHistogram, but we need it for tests.
constexpr bool operator!=(const CustomType& lhs, const CustomType& rhs) noexcept
{
  return lhs.value != rhs.value;
}

// std::ostream support for CustomType.
// This is not needed for CumulativeHistogram, but we need it for tests.
std::ostream& operator<< (std::ostream& stream, const CustomType& object)
{
  return stream << "CustomType{ " << object.value << "}";
}

class CustomSum
{
public:
  // Not default-constructible.
  constexpr explicit CustomSum(int _) noexcept {}

  constexpr CustomType operator()(const CustomType& lhs, const CustomType& rhs) const noexcept
  {
    return CustomType{ lhs.value ^ rhs.value };
  }
};

TEST(CumulativeHistogram, UserDefinedType)
{
  CumulativeHistogram<CustomType, CustomSum> histogram(CustomSum(10));
  histogram.reserve(10);
  histogram.resize(2);
  histogram.pushBack(CustomType{ 5 });
  histogram.pushBack(CustomType{ 1 });
  histogram.pushBack(CustomType{ 6 });
  histogram.pushBack(CustomType{ 2 });
  histogram.increment(0, CustomType{ 8 });  // element(0) := element(0) XOR 8;
  histogram.popBack();
  EXPECT_TRUE(CheckPrefixSums(histogram));
  CumulativeHistogram<CustomType, CustomSum> histogram2 = histogram;
  EXPECT_TRUE(CheckPrefixSums(histogram2));
  CumulativeHistogram<CustomType, CustomSum> histogram3 = std::move(histogram2);
  EXPECT_TRUE(CheckPrefixSums(histogram3));
  histogram3.clear();
  histogram = histogram3;
}

}
}