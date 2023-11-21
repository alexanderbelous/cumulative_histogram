#include <cumulative_histogram/CumulativeHistogram.h>

#include <benchmark/benchmark.h>

#include <chrono>
#include <cstdint>
#include <numeric>
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

using ElementTypeForBenchmark = std::uint32_t;

void BM_CumulativeHisogramBuildTree (benchmark::State& state) {
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) {
    CumulativeHistogram<ElementTypeForBenchmark> histogram(num_elements, 1);
    benchmark::DoNotOptimize(histogram);
  }
}
BENCHMARK(BM_CumulativeHisogramBuildTree)->Range(8, 256 << 10);

void BM_ArrayOfPrefixSumsIncrement(benchmark::State& state) {
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  ArrayOfPrefixSums<ElementTypeForBenchmark> histogram(num_elements);
  std::size_t i = 0;
  for (auto _ : state) {
    histogram.increment(i, 1);
    ++i;
    if (i == num_elements) {
      i = 0;
    }
  }
}
BENCHMARK(BM_ArrayOfPrefixSumsIncrement)->Range(8, 256 << 10);

void BM_CumulativeHisogramIncrement(benchmark::State& state) {
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  CumulativeHistogram<ElementTypeForBenchmark> histogram(num_elements);
  std::size_t i = 0;
  for (auto _ : state) {
    histogram.increment(i, 1);
    ++i;
    if (i == num_elements) {
      i = 0;
    }
  }
}
BENCHMARK(BM_CumulativeHisogramIncrement)->Range(8, 256 << 10);

void BM_ArrayOfElementsPrefixSum(benchmark::State& state) {
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  ArrayOfElements<ElementTypeForBenchmark> histogram(num_elements);
  std::size_t i = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(histogram.prefixSum(i));
    ++i;
    if (i == num_elements) {
      i = 0;
    }
  }
}
BENCHMARK(BM_ArrayOfElementsPrefixSum)->RangeMultiplier(2)->Range(8, 256 << 10);

void BM_CumulativeHisogramPrefixSum(benchmark::State& state) {
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  CumulativeHistogram<ElementTypeForBenchmark> histogram(num_elements);
  std::size_t i = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(histogram.prefixSum(i));
    ++i;
    if (i == num_elements) {
      i = 0;
    }
  }
}
BENCHMARK(BM_CumulativeHisogramPrefixSum)->RangeMultiplier(2)->Range(8, 256 << 10);

void BM_CumulativeHisogramTotalSum(benchmark::State& state) {
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  CumulativeHistogram<ElementTypeForBenchmark> histogram(num_elements);
  for (auto _ : state) {
    benchmark::DoNotOptimize(histogram.totalSum());
  }
}
BENCHMARK(BM_CumulativeHisogramTotalSum)->RangeMultiplier(2)->Range(8, 256 << 10);

void BM_ArrayOfPrefixSumsLowerBound(benchmark::State& state) {
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  ArrayOfPrefixSums<ElementTypeForBenchmark> histogram(num_elements, 1);
  std::size_t value = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(histogram.lowerBound(static_cast<ElementTypeForBenchmark>(value)));
    ++value;
    if (value == num_elements + 1)
    {
      value = 0;
    }
  }
}
BENCHMARK(BM_ArrayOfPrefixSumsLowerBound)->Range(8, 256 << 10);

void BM_CumulativeHisogramLowerBound(benchmark::State& state) {
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  CumulativeHistogram<ElementTypeForBenchmark> histogram(num_elements, 1);
  std::size_t value = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(histogram.lowerBound(static_cast<ElementTypeForBenchmark>(value)));
    ++value;
    if (value == num_elements + 1)
    {
      value = 0;
    }
  }
}
BENCHMARK(BM_CumulativeHisogramLowerBound)->Range(8, 256 << 10);

// Measures the average time of calling pushBack() N times for a CumulativeHistogram
// currently storing N elements, and capable of storing 2N elements.
// The purpose of this benchmark is to demonstrate that the time complexity is amortized constant.
void BM_CumulativeHisogramPushBackNoReallocation(benchmark::State& state) {
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  CumulativeHistogram<ElementTypeForBenchmark> histogram(num_elements, 1);
  histogram.reserve(num_elements * 2);
  for (auto _ : state) {
    const auto start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < num_elements; ++i) {
      histogram.pushBack(static_cast<ElementTypeForBenchmark>(1));
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
