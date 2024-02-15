#pragma once

#include <cumulative_histogram/Math.h>

#include <cassert>
#include <span>

namespace CumulativeHistogram_NS::Detail_NS
{

// Non-template base for FullTreeView.
//
// CumulativeHistogram stores auxiliary counters for sums of certain elements. These counters are updated
// whenever the respective elements are modified - because of this, the time complexity of
// CumulativeHistogram::increment() is O(logN) instead of O(1). However, having these sums precomputed also
// allows to compute any prefix sum in O(logN).
//
// CumulativeHistogram splits the elements into buckets of size B: the first B elements go into the 0th
// bucket, the next B elements go into the 1st bucket, and so on. These buckets are stored in an implicit
// binary tree
// Let N be the number of elements. The number of buckets is M = ceil(N/B); each bucket (except, possibly,
// the last one) stores exactly B elements.
// The tree is constructed as descibed below:
//   * If M <= 1, then the tree has no nodes.
//   * Otherwise, a node is constructed for the root.
//     * The left subtree is constructed for the first ceil(M/2) buckets.
//     * The right subtree is constructed for the remaining floor(M/2) buckets.
//     * The root itself stores the sum of all elements from the left subtree.
//
// The nodes of the tree are stored in a plain array:
//   { root, left0, left1, ..., leftM1, right0, right1, ..., rightM2 }
// Due to this representation, any subtree can be viewed as a subspan of the nodes array.
//
// For example, let N = 13, B = 2. Then the number of buckets M = ceil(N/B) = ceil(13/2) = 7,
// and the tree looks like this:
//         n0
//       /   \
//     n1     n4
//    / \    /
//   n2 n3  n5
//
// Let b[i] be the sum of elements from the i-th bucket. Then, the values stored in the nodes are:
// +-------------+-------+----+----+-------+----+
// | n0          | n1    | n2 | n3 | n4    | n5 |
// +-------------+-------+----+----+-------+----+
// | b0+b1+b2+b3 | b0+b1 | b0 | b2 | b4+b5 | b4 |
// +-------------+-------+----+----+-------+----+
// The sum of the first K buckets sB(K) can be computed via O(logM) additions for any 0 <= K < M:
// +-------+-------+-------+-------+-------+-------+-------+
// | sB(0) | sB(1) | sB(2) | sB(3) | sB(4) | sB(5) | sB(6) |
// +-------+-------+-------+-------+-------+-------+-------+
// | 0     | n2    | n1    | n1+n3 | n0    | n0+n5 | n0+n4 |
// +-------+-------+-------+-------+-------+-------+-------+
// With this, we can compute any prefix sum s(i), 0 <= i < N as:
//     // Index of the bucket that element i belongs to.
//     K = floor(i/B);
//     // Sum of elements from this bucket.
//     sum_b = element[K*B] + element[K*B + 1] + ... + element[i];
//     // Sum of elements from the previous buckets.
//     sum_a = sB(K);
       // Prefix sum s(i) = element[0] + element[1] + ... + element[i].
//     s(i) = sum_a + sum_b;
// The time complexity of this algorithm is O(logM) + O(B) = O(log(N/B)) + O(B).
// B is a constant independent of N, so the overall time complexity is O(logN).
//
// CumulativeHistogram also allows reserving memory for future elements, in which case the tree is
// constructed for Nmax elements, where Nmax is the desired capacity (0 <= N <= Nmax). However, this doesn't
// affect the time complexity of any operation: if Nmax is much greater than N, we can easily determine the
// smallest leftmost subtree that represents all N elements, and traverse it instead of traversing the full
// tree.
//
// This class provides an API for traversing a tree constructed for the specified number of buckets. It is
// not meant to be used directly - instead, the template class FullTreeView (which is derived from
// FullTreeViewBase) should be used.
class FullTreeViewBase
{
public:
  // Constructs a view for a tree for the specified number of buckets.
  // \param num_buckets - the number of buckets that the tree represents.
  constexpr explicit FullTreeViewBase(std::size_t num_buckets) noexcept:
    bucket_first_(0),
    num_buckets_(num_buckets)
  {}

  // Returns true if the tree has no nodes, false otherwise.
  constexpr bool empty() const noexcept
  {
    // Same as numNodes() == 0
    return num_buckets_ <= 1;
  }

  constexpr std::size_t numNodes() const noexcept
  {
    return countNodesInBucketizedTree(num_buckets_);
  }

  constexpr std::size_t bucketFirst() const noexcept
  {
    return bucket_first_;
  }

  constexpr std::size_t numBuckets() const noexcept
  {
    return num_buckets_;
  }

  // Returns 0-based index of the first bucket (inclusive) of the right subtree.
  constexpr std::size_t pivot() const noexcept
  {
    return bucket_first_ + (num_buckets_ + 1) / 2;
  }

protected:
  // Switches to the immediate left subtree.
  // \return the offset of the root node of the left subtree from the root node of the parent tree.
  constexpr std::size_t switchToLeftChild() noexcept
  {
    // assert(!empty())
    ++num_buckets_ >>= 1;  // num_buckets_ = ceil(num_buckets_ / 2)
    return 1;
  }

  // Switches to the immediate right subtree.
  // \return the offset of the root node of the right subtree from the root node of the parent tree.
  constexpr std::size_t switchToRightChild() noexcept
  {
    // assert(!empty())
    const std::size_t num_buckets_left = (num_buckets_ + 1) >> 1;  // ceil(num_buckets_ / 2)
    bucket_first_ += num_buckets_left;
    num_buckets_ >>= 1;  // num_buckets_ = floor(num_buckets_ / 2)
    // Skip root and nodes of the left subtree.
    // Same as 1 + countNodesInBucketizedTree(num_buckets_left);
    return num_buckets_left;
  }

  // Switches to the leftmost subtree at the specified depth.
  // \return the offset of the root node of the new tree from the root node of the original tree.
  constexpr std::size_t switchToLeftmostChild(std::size_t level) noexcept
  {
    assert(level == 0 || num_buckets_ > (static_cast<std::size_t>(1) << (level - 1)));
    num_buckets_ = countElementsInLeftmostSubtree(num_buckets_, level);
    return level;
  }

  // Switches to the rightmost subtree at the specified depth.
  // \return the offset of the root node of the new tree from the root node of the original tree.
  constexpr std::size_t switchToRightmostChild(std::size_t level) noexcept
  {
    const std::size_t num_buckets_new = num_buckets_ >> level;  // floor(N / 2^k)
    assert(num_buckets_new >= 1);

    const std::size_t num_buckets_skipped = num_buckets_ - num_buckets_new;
    // By definition, the nodes of the rightmost subtree are the last nodes of the array.
    // Therefore, in order to compute the offset of the new root node from the old root node,
    // we can simply compute the difference between the number of nodes of the old tree and the new tree.
    // Since both of these trees represent at least 1 bucket, the number of their nodes is given by:
    //   const std::size_t num_nodes_old = num_buckets_ - 1;
    //   const std::size_t num_nodes_new = num_buckets_new - 1;
    // Hence, the offset of the new root node from the old one is:
    //   const std::size_t root_offset = num_buckets_ - num_buckets_new;
    // Which is equal to num_buckets_skipped.

    bucket_first_ += num_buckets_skipped;
    num_buckets_ = num_buckets_new;
    return num_buckets_skipped;
  }

private:
  // Index of the first bucket represented by the tree.
  std::size_t bucket_first_;
  // The number of buckets represented by the tree.
  std::size_t num_buckets_;
};

// High-level API for interacing with the implicit tree data structure.
// Note that FullTreeView is unaware of the fact that some nodes may be inactive - it simply provides a way
// to traverse the tree. While it's possible to design an API that would skip inactive nodes, and it might
// even perform better in the best case (i.e., traversing from the root to a leaf may take O(1) in the best
// case, while for FullTreeView it's always O(logN)), FullTreeView will still perform better on average,
// because the constant factor will be smaller.
template<class T>
class FullTreeView : public FullTreeViewBase
{
public:
  constexpr FullTreeView(T* root, std::size_t num_buckets) noexcept:
    FullTreeViewBase(num_buckets),
    root_(root)
  {}

  constexpr std::span<T> nodes() const noexcept
  {
    return std::span<T>{root_, numNodes()};
  }

  constexpr T& root() const
  {
    return *root_;
  }

  // Switches to the immediate left subtree.
  constexpr void switchToLeftChild() noexcept
  {
    root_ += FullTreeViewBase::switchToLeftChild();
  }

  // Switches to the immediate right subtree.
  constexpr void switchToRightChild() noexcept
  {
    root_ += FullTreeViewBase::switchToRightChild();
  }

  // Returns the immediate left subtree.
  constexpr FullTreeView leftChild() const noexcept
  {
    FullTreeView tree = *this;
    tree.switchToLeftChild();
    return tree;
  }

  // Returns the immediate right subtree.
  constexpr FullTreeView rightChild() const noexcept
  {
    FullTreeView tree = *this;
    tree.switchToRightChild();
    return tree;
  }

private:
  // Root of the tree.
  T* root_;
};

// Constructs a FullTreeView for the currently effective tree representing the given elements.
// \param num_elements - the number of elements.
// \param capacity - the maximum number of elements that the tree can represent.
// \param bucket_size - the number of elements per bucket.
// \param nodes - all nodes of the tree.
// \return FullTreeView for the currently effective tree representing all `num_elements`.
// Time complexity: O(1).
template<class T>
constexpr FullTreeView<T> makeFullTreeView(std::size_t num_elements, std::size_t capacity,
                                           std::size_t bucket_size, std::span<T> nodes) noexcept
{
  assert(num_elements <= capacity);
  const std::size_t num_buckets = countBuckets(num_elements, bucket_size);
  const std::size_t bucket_capacity = countBuckets(capacity, bucket_size);
  assert(nodes.size() == countNodesInBucketizedTree(bucket_capacity));
  const std::size_t root_level = findDeepestNodeForElements(num_buckets, bucket_capacity);
  const std::size_t bucket_capacity_at_level = countElementsInLeftmostSubtree(bucket_capacity, root_level);
  return FullTreeView<T> { nodes.data() + root_level, bucket_capacity_at_level };
}

}  // namespace CumulativeHistogram_NS::Detail_NS
