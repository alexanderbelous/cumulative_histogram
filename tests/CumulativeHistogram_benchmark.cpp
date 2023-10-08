#include <cumulative_histogram/CumulativeHistogram.h>

#include <benchmark/benchmark.h>

#include <chrono>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using ::CumulativeHistogram_NS::CumulativeHistogram;

namespace {

template<class T>
class ArrayOfElements {
 public:
  explicit ArrayOfElements(std::size_t num_elements):
    data_(num_elements)
  {}

  explicit ArrayOfElements(std::size_t num_elements, const T& value) :
    data_(num_elements, value)
  {}

  std::size_t size() const noexcept {
    return data_.size();
  }

  void increment(std::size_t k, const T& value) {
    if (k >= size()) {
      throw std::out_of_range("Index is out of range");
    }
    data_[k] += value;
  }

  T prefixSum(std::size_t k) const {
    if (k >= size()) {
      throw std::out_of_range("Index is out of range");
    }
    return std::accumulate(data_.begin(), data_.begin() + k + 1, T{});
  }

 private:
  std::vector<T> data_;
};

template<class T>
class ArrayOfPrefixSums {
 public:
  using const_iterator = typename std::vector<T>::const_iterator;

  explicit ArrayOfPrefixSums(std::size_t num_elements):
    data_(num_elements)
  {}

  explicit ArrayOfPrefixSums(std::size_t num_elements, const T& value) :
    data_(num_elements, value)
  {}

  std::size_t size() const noexcept {
    return data_.size();
  }

  void increment(std::size_t k, const T& value) {
    if (k >= size()) {
      throw std::out_of_range("Index is out of range");
    }
    for (auto iter = data_.begin() + k; iter != data_.end(); ++iter) {
      *iter += value;
    }
  }

  T prefixSum(std::size_t k) const {
    return data_.at(k);
  }

  std::pair<const_iterator, T> lowerBound(const T& value) const {
    auto iter = std::lower_bound(data_.begin(), data_.end(), value);
    if (iter == data_.end()) {
      return { iter, T{} };
    }
    return { iter, *iter };
  }

 private:
  std::vector<T> data_;
};

void BM_CumulativeHisogramBuildTree (benchmark::State& state) {
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) {
    CumulativeHistogram<std::uint32_t> histogram(num_elements, 1);
    benchmark::DoNotOptimize(histogram);
  }
}
BENCHMARK(BM_CumulativeHisogramBuildTree)->Range(8, 256 << 10);

void BM_ArrayOfPrefixSumsIncrement(benchmark::State& state) {
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  ArrayOfPrefixSums<std::uint32_t> histogram(num_elements);
  std::mt19937 gen;  // mersenne_twister_engine seeded with some default value.
  std::uniform_int_distribution<std::size_t> distribution{ 0, num_elements - 1 };
  for (auto _ : state) {
    // Increment a random element by 1.
    const std::size_t i = distribution(gen);
    histogram.increment(i, 1);
  }
}
BENCHMARK(BM_ArrayOfPrefixSumsIncrement)->Range(8, 256 << 10);

void BM_CumulativeHisogramIncrement(benchmark::State& state) {
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  CumulativeHistogram<std::uint32_t> histogram(num_elements);
  std::mt19937 gen;  // mersenne_twister_engine seeded with some default value.
  std::uniform_int_distribution<std::size_t> distribution{ 0, num_elements - 1 };
  for (auto _ : state) {
    // Increment a random element by 1.
    const std::size_t i = distribution(gen);
    histogram.increment(i, 1);
  }
}
BENCHMARK(BM_CumulativeHisogramIncrement)->Range(8, 256 << 10);

void BM_ArrayOfElementsPrefixSum(benchmark::State& state) {
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  ArrayOfElements<std::uint32_t> histogram(num_elements);
  std::mt19937 gen;  // mersenne_twister_engine seeded with some default value.
  std::uniform_int_distribution<std::size_t> distribution{ 0, num_elements - 1 };
  for (auto _ : state) {
    // Compute the i-th prefix sum for a random i.
    const std::size_t i = distribution(gen);
    benchmark::DoNotOptimize(histogram.prefixSum(i));
  }
}
BENCHMARK(BM_ArrayOfElementsPrefixSum)->RangeMultiplier(2)->Range(8, 256 << 10);

void BM_CumulativeHisogramPrefixSum(benchmark::State& state) {
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  CumulativeHistogram<std::uint32_t> histogram(num_elements);
  std::mt19937 gen;  // mersenne_twister_engine seeded with some default value.
  std::uniform_int_distribution<std::size_t> distribution{ 0, num_elements - 1 };
  for (auto _ : state) {
    // Compute the i-th prefix sum for a random i.
    const std::size_t i = distribution(gen);
    benchmark::DoNotOptimize(histogram.prefixSum(i));
  }
}
BENCHMARK(BM_CumulativeHisogramPrefixSum)->RangeMultiplier(2)->Range(8, 256 << 10);

void BM_ArrayOfPrefixSumsLowerBound(benchmark::State& state) {
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  ArrayOfPrefixSums<std::uint32_t> histogram(num_elements, 1);
  std::mt19937 gen;  // mersenne_twister_engine seeded with some default value.
  std::uniform_int_distribution<std::uint32_t> distribution{ 0, static_cast<std::uint32_t>(num_elements + 1) };
  for (auto _ : state) {
    // Compute the lower bound for a random value i.
    const std::uint32_t value = distribution(gen);
    benchmark::DoNotOptimize(histogram.lowerBound(value));
  }
}
BENCHMARK(BM_ArrayOfPrefixSumsLowerBound)->Range(8, 256 << 10);

void BM_CumulativeHisogramLowerBound(benchmark::State& state) {
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  CumulativeHistogram<std::uint32_t> histogram(num_elements, 1);
  std::mt19937 gen;  // mersenne_twister_engine seeded with some default value.
  std::uniform_int_distribution<std::uint32_t> distribution{ 0, static_cast<std::uint32_t>(num_elements + 1) };
  for (auto _ : state) {
    // Compute the lower bound for a random value.
    const std::uint32_t value = distribution(gen);
    benchmark::DoNotOptimize(histogram.lowerBound(value));
  }
}
BENCHMARK(BM_CumulativeHisogramLowerBound)->Range(8, 256 << 10);

// Measures the average time of calling push_back() N times for a CumulativeHistogram
// currently storing N elements, and capable of storing 2N elements.
// The purpose of this benchmark is to demonstrate that the time complexity is amortized constant.
void BM_CumulativeHisogramPushBackNoReallocation(benchmark::State& state) {
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  CumulativeHistogram<std::uint32_t> histogram(num_elements, 1);
  histogram.reserve(num_elements * 2);
  for (auto _ : state) {
    const auto start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < num_elements; ++i) {
      histogram.push_back(1);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count() / num_elements);
    // Reset the number of elements to N.
    histogram.resize(num_elements);
  }
}
BENCHMARK(BM_CumulativeHisogramPushBackNoReallocation)->Range(8, 256 << 10)->Iterations(100)->UseManualTime();

}

BENCHMARK_MAIN();
