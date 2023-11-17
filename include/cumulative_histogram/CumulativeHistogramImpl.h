#pragma once

#include <bit>
#include <cstddef>

namespace CumulativeHistogram_NS::Detail_NS
{

// Returns floor(log2(x)).
// The result is unspecified if value == 0.
constexpr std::size_t floorLog2(std::size_t value) noexcept
{
  return std::bit_width(value) - 1;
}

// Returns ceil(log2(x)).
// The result is unspecified if value == 0.
constexpr std::size_t ceilLog2(std::size_t value) noexcept
{
  const bool is_power_of_2 = std::has_single_bit(value);
  const std::size_t floor_log2 = floorLog2(value);
  return is_power_of_2 ? floor_log2 : (floor_log2 + 1);
}

// Returns the number of buckets needed to represent the specified number of elements.
// \param num_elements - the number of elements.
// \param bucket_size - the number of elements per bucket.
// \returns ceil(num_elements / bucket_size).
// The behavior is undefined if bucket_size == 0.
constexpr std::size_t countBuckets(std::size_t num_elements, std::size_t bucket_size) noexcept
{
  return (num_elements / bucket_size) + (num_elements % bucket_size != 0);
}

// Returns the number of nodes in a tree representing the specified number of buckets.
// \param num_buckets - the number of buckets represented by the tree.
constexpr std::size_t countNodesInBucketizedTree(std::size_t num_buckets) noexcept
{
  // f(0) = 0
  // f(1) = 0
  // for any N > 1:
  //   f(2N) = 1 + 2*f(N)
  //   f(2N+1) = 1 + f(N+1) + f(N)
  // It's easy to prove by induction that f(N) = N - 1 for any N > 1.
  return num_buckets == 0 ? 0 : (num_buckets - 1);
}

// Returns the number of elements represented by the leftmost subtree with root at the specified level.
// \param num_elements - the total number of elements represented by the tree.
// \param level - depth of the tree for which to count the number of elements. If level == 0, the
//                function returns `num_elements`.
// Time complexity: O(1).
constexpr std::size_t countElementsInLeftmostSubtree(std::size_t num_elements, std::size_t level) noexcept
{
  // First, note that h(x) = ceil(ceil(x/2)/2) = ceil(x/4). Proof:
  //   h(4a) = a
  //   h(4a+1) = ceil(ceil((4a+1)/2)/2) = ceil((2a+1)/2) = a+1
  //   h(4a+2) = ceil(ceil((4a+2)/2)/2) = ceil((2a+1)/2) = a+1
  //   h(4a+3) = ceil(ceil((4a+3)/2)/2) = ceil((2a+2)/2) = a+1
  //
  // The number of elements in the leftmost subtree is computed as:
  //   f(0) = N = ceil(N/1)
  //   f(1) = ceil(f(0)/2) = ceil(N/2)
  //   f(2) = ceil(f(1)/2) = ceil(ceil(N/2)/2) = ceil(N/4)
  //   f(3) = ceil(f(2)/2) = ceil(ceil(N/4)/2) = ceil(N/8)
  //   f(k) = ceil(N/2^k)
  const std::size_t mask = (static_cast<std::size_t>(1) << level) - 1;
  const std::size_t floored_result = num_elements >> level;
  const std::size_t remainder = num_elements & mask;
  return floored_result + (remainder != 0);
}

// Finds the deepest node containing all of the given elements in the tree with the specified capacity.
// \param num_elements - specifies the range [0; num_elements) that must be contained by the node.
// \param capacity - the total number of elements that can be represented by the tree.
// \return the depth of the deepest node containing the elements [0; num_elements)
//         in the optimal tree representing the elements [0; capacity),
//         or static_cast<std::size_t>(-1) if capacity < num_elements.
//         The depth of the root is 0.
// Time complexity: O(1).
constexpr std::size_t findDeepestNodeForElements(std::size_t num_elements, std::size_t capacity) noexcept
{
  // Let's assume that we have an optimal tree representing Nmax elements.
  // Its left subtree represents ceil(Nmax/2) elements.
  // The left subtree of the above-mentioned subtree represents ceil(ceil(Nmax/2)/2) = ceil(Nmax/4) elements.
  // and so on.
  //   x0 = Nmax, x1 = ceil(Nmax/2), ..., xK = ceil(Nmax/2^K)
  // We want to find the deepest subtree that represents at least N elements.
  // Which brings us to the inequality:
  //   ceil(Nmax/2^k) >= N
  // <=>
  //   Nmax/2^k + 1 > N
  // <=>
  //   Nmax/(N-1) > 2^k
  // <=>
  //   k < log2(Nmax/(N-1))
  // <=>
  //   The greatest such k is ceil(log2(Nmax/(N-1)))-1
  //
  // Edge cases:
  // * Nmax=0 is an edge case because log2(0) is undefined. In our case it means that
  //   the capacity of CumulativeHistogram is 0, so we return 0.
  // * Nmax=1 is an edge case because in this case N can be either 0 or 1 - both of
  //   which are edge cases (see below). If CumulativeHistogram can store 1 element
  //   at most, then the tree has 0 nodes, so we return 0.
  // * N=1 is an edge case because it causes division by 0. In our case it means that
  //   we want to find the deepest leftmost subtree that represents at least 1 element.
  //   Note that due to ceiling the formula above implies that k should be infinite.
  //   That's not what we want though - if N=1, then we want to simply find the deepest
  //   leftmost subtree (i.e. the first leftmost subtree that represents exactly 1 element).
  // * N=0 is an edge case because it will result into log2(-Nmax), which is undefined.
  //   In our case it means that we should return the deepest leftmost subtree, which is the
  //   same as calling findDeepestNodeForElements(1, Nmax).
  if (capacity < 2) return 0;
  if (num_elements < 2)
  {
    // Find the smallest k such that ceil(Nmax/2^k) == 1, which is simply ceil(log2(Nmax))
    return ceilLog2(capacity);
  }
  // Note that ceil(log2(x)) = ceil(log2(ceil(x)).
  const std::size_t floored_ratio = capacity / (num_elements - 1);
  const std::size_t remainder = capacity % (num_elements - 1);
  const std::size_t ratio = floored_ratio + (remainder != 0);  // ceil(Nmax/(N-1))
  return ceilLog2(ratio) - 1;
}

// Find the depth of the leftmost subtree that represents exactly the specified number of buckets.
// \param capacity_to_find - bucket capacity of the subtree to find.
// \param capacity - bucket capacity of the full tree.
// \return depth of the leftmost subtree that represents exactly `capacity_to_find` buckets,
//         or static_cast<std::size_t>(-1) if there is no such subtree.
//         Note that if capacity > 0, then no subtree of the full tree represents exactly 0 buckets,
//         so the function will return -1. However, if both capacity and capacity_to_find are 0, then
//         the function returns 0.
// Time complexity: O(1).
constexpr std::size_t findLeftmostSubtreeWithExactCapacity(std::size_t capacity_to_find,
                                                           std::size_t capacity) noexcept
{
  // Edge case: a smaller tree cannot contain a larger tree.
  if (capacity < capacity_to_find)
  {
    return static_cast<std::size_t>(-1);
  }
  // Find the deepest leftmost node of the new tree that contains at least `capacity_to_find` buckets.
  const std::size_t level = findDeepestNodeForElements(capacity_to_find, capacity);
  // Get the number of elements that this node contains.
  const std::size_t num_elements_at_level = countElementsInLeftmostSubtree(capacity, level);
  // The old tree is a subtree of the new one if and only if the number of elements matches exactly.
  if (num_elements_at_level == capacity_to_find)
  {
    return level;
  }
  return static_cast<std::size_t>(-1);
}

}  // namespace CumulativeHistogram_NS::Detail_NS
