#include <cumulative_histogram/CumulativeHistogram.h>

#include <benchmark/benchmark.h>

#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

using ::CumulativeHistogram_NS::CumulativeHistogram;

namespace {

template<class T>
class ArrayOfPrefixSums {
 public:
  explicit ArrayOfPrefixSums(std::size_t num_elements):
    data_(num_elements)
  {}

  explicit ArrayOfPrefixSums(std::size_t num_elements, const T& value) :
    data_(num_elements, value)
  {}

  std::size_t size() noexcept {
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

 private:
  std::vector<T> data_;
};

void BM_ArrayOfPrefixSumsRandomIncrementAndPrefixSum(benchmark::State& state) {
  constexpr std::size_t kNumOperations = 1024;
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  ArrayOfPrefixSums<std::uint32_t> histogram(num_elements);
  std::mt19937 gen;  // mersenne_twister_engine seeded with some default value.
  std::uniform_int_distribution<std::size_t> distribution{ 0, num_elements - 1 };
  for (auto _ : state)
    for (std::size_t iteration = 0; iteration < kNumOperations; ++iteration) {
      const std::size_t i = distribution(gen);
      histogram.increment(i, 1);
      const std::uint32_t value = histogram.prefixSum(i);
      benchmark::DoNotOptimize(value);
    }
}
BENCHMARK(BM_ArrayOfPrefixSumsRandomIncrementAndPrefixSum)->Range(8, 32 << 10);

void BM_CumulativeHisogramRandomIncrementAndPrefixSum(benchmark::State& state) {
  constexpr std::size_t kNumOperations = 1024;
  const std::size_t num_elements = static_cast<std::size_t>(state.range(0));
  CumulativeHistogram<std::uint32_t> histogram(num_elements);
  std::mt19937 gen;  // mersenne_twister_engine seeded with some default value.
  std::uniform_int_distribution<std::size_t> distribution{ 0, num_elements - 1 };
  for (auto _ : state)
    for (std::size_t iteration = 0; iteration < kNumOperations; ++iteration) {
      const std::size_t i = distribution(gen);
      histogram.increment(i, 1);
      const std::uint32_t value = histogram.prefixSum(i);
      benchmark::DoNotOptimize(value);
    }
}
BENCHMARK(BM_CumulativeHisogramRandomIncrementAndPrefixSum)->Range(8, 32 << 10);

}

BENCHMARK_MAIN();
