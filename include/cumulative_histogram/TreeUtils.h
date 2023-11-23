#pragma once

#include <cumulative_histogram/FullTreeView.h>
#include <cumulative_histogram/Math.h>

#include <cassert>
#include <functional>
#include <numeric>
#include <span>
#include <type_traits>
#include <utility>

namespace CumulativeHistogram_NS::Detail_NS
{

  template<class T, class SumOperation>
  class TreeBuilder
  {
  public:
    // Construct a TreeBuilder for the given elements.
    // \param elements - all elements represented by the main tree.
    // \param bucket_size - the number of elements per bucket.
    // \param sum_op - function object that implements addition for the type T.
    constexpr TreeBuilder(std::span<const T> elements, std::size_t bucket_size, SumOperation sum_op) noexcept;

    // Initializes the active nodes of the given tree according to the elements.
    // A node is active if there are elements in both its left and right subtrees.
    // \param tree - some subtree of the main tree to build.
    // \return the sum of all elements represented by `tree`.
    // Time complexity: O(M), where M is the number of active elements represented by `tree`.
    constexpr T build(const FullTreeView<T>& tree) const;

  private:
    std::span<const T> elements_;
    std::size_t bucket_size_;
    // Total number of active buckets.
    std::size_t num_buckets_;
    // Function object that implements addition for the type T.
#ifdef _MSC_VER
    [[msvc::no_unique_address]] SumOperation sum_op_;
#else
    [[no_unique_address]] SumOperation sum_op_;
#endif
  };

  template<class T, class SumOperation>
  constexpr TreeBuilder<T, SumOperation>::TreeBuilder(std::span<const T> elements, std::size_t bucket_size,
    SumOperation sum_op) noexcept :
    elements_(elements),
    bucket_size_(bucket_size),
    num_buckets_(countBuckets(elements.size(), bucket_size)),
    sum_op_(sum_op)
  {}

  template<class T, class SumOperation>
  constexpr T TreeBuilder<T, SumOperation>::build(const FullTreeView<T>& tree) const
  {
    assert(tree.numBuckets() > 0);
    assert(tree.bucketFirst() < num_buckets_);
    FullTreeView<T> t = tree;
    T total_sum{};
    while (!t.empty())
    {
      // Just switch to the left subtree if the root is inactive.
      if (num_buckets_ <= t.pivot())
      {
        t.switchToLeftChild();
        continue;
      }
      T total_sum_left = build(t.leftChild());
      if constexpr (std::is_arithmetic_v<T> && std::is_same_v<SumOperation, std::plus<>>)
      {
        total_sum += total_sum_left;
      }
      else
      {
        total_sum = sum_op_(std::move(total_sum), total_sum_left);
      }
      t.root() = std::move(total_sum_left);
      t.switchToRightChild();
    }
    // Add elements from the last active bucket represented by the input tree.
    const std::size_t element_first = t.bucketFirst() * bucket_size_;
    const std::size_t num_elements = std::min(elements_.size() - element_first, bucket_size_);
    const std::span<const T> elements_in_bucket = elements_.subspan(element_first, num_elements);
    return std::accumulate(elements_in_bucket.begin(), elements_in_bucket.end(), std::move(total_sum), sum_op_);
  }

  // Initializes the active nodes of the given tree according to the elements.
  // A node is active if there are elements in both its left and right subtrees.
  // \param elements - all elements represented by the main tree.
  // \param tree - some subtree of the main tree to build.
  // \param bucket_size - the number of elements per bucket.
  // \param sum_op - function object that implements addition for the type T.
  // \return the sum of all elements represented by `tree`.
  // Time complexity: O(M), where M is the number of elements from `elements` represented by `tree`
  //                  (M <= elements.size()).
  template<class T, class SumOperation>
  T buildBucketizedTree(std::span<const T> elements, const FullTreeView<T>& tree,
                        std::size_t bucket_size, SumOperation sum_op)
  {
    return TreeBuilder<T, SumOperation>(elements, bucket_size, sum_op).build(tree);
  }

  // Computes the total sum of elements of a tree which is at its full capacity.
  // \param elements - elements to sum.
  // \param tree - auxiliary tree for `elements`.
  // \param bucket_size - the number of elements per bucket.
  // \param sum_op - function object that implements addition for the type T.
  // The behavior is undefined if
  //   elements.empty() || elements.size() != tree.numBuckets() * bucket_size
  // \returns the total sum of elements from `elements`.
  // Time complexity: O(log(N/bucket_size) + bucket_size), where N = elements.size().
  template<class T, class SumOperation>
  constexpr T sumElementsOfFullTree(std::span<const T> elements, FullTreeView<const T> tree,
                                    std::size_t bucket_size, SumOperation sum_op)
  {
    assert(!elements.empty());
    assert(elements.size() == tree.numBuckets() * bucket_size);
    T result{};
    while (!tree.empty())
    {
      if constexpr (std::is_arithmetic_v<T> && std::is_same_v<SumOperation, std::plus<>>)
      {
        result += tree.root();
      }
      else
      {
        result = sum_op(std::move(result), tree.root());
      }
      tree.switchToRightChild();
    }
    // Add elements from the last bucket.
    const std::span<const T> last_bucket = elements.last(bucket_size);
    return std::accumulate(last_bucket.begin(), last_bucket.end(), std::move(result), sum_op);
  }
}
