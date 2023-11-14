#pragma once

#include <cumulative_histogram/CumulativeHistogramImpl.h>

#include <cassert>
#include <span>

namespace CumulativeHistogram_NS
{
namespace Detail_NS
{
  // Non-template base for FullTreeView.
  class FullTreeViewBase {
   public:
    constexpr explicit FullTreeViewBase(std::size_t num_buckets) noexcept:
      bucket_first_(0),
      num_buckets_(num_buckets)
    {}

    // Returns true if the tree has no nodes, false otherwise.
    constexpr bool empty() const noexcept {
      // Same as numNodes() == 0
      return num_buckets_ <= 1;
    }

    constexpr std::size_t numNodes() const noexcept {
      return countNodesInBucketizedTree(num_buckets_);
    }

    constexpr std::size_t bucketFirst() const noexcept {
      return bucket_first_;
    }

    constexpr std::size_t numBuckets() const noexcept {
      return num_buckets_;
    }

    // Returns 0-based index of the first bucket (inclusive) of the right subtree.
    constexpr std::size_t pivot() const noexcept {
      return bucket_first_ + (num_buckets_ + 1) / 2;
    }

   protected:
    // Switches to the immediate left subtree.
    // \return the offset of the root node of the left subtree from the root node of the parent tree.
    constexpr std::size_t switchToLeftChild() noexcept {
      // assert(!empty())
      ++num_buckets_ >>= 1;  // num_buckets_ = ceil(num_buckets_ / 2)
      return 1;
    }

    // Switches to the immediate right subtree.
    // \return the offset of the root node of the right subtree from the root node of the parent tree.
    constexpr std::size_t switchToRightChild() noexcept {
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
    constexpr std::size_t switchToLeftmostChild(std::size_t level) noexcept {
      assert(level == 0 || num_buckets_ > (static_cast<std::size_t>(1) << (level - 1)));
      num_buckets_ = countElementsInLeftmostSubtree(num_buckets_, level);
      return level;
    }

    // Switches to the rightmost subtree at the specified depth.
    // \return the offset of the root node of the new tree from the root node of the original tree.
    constexpr std::size_t switchToRightmostChild(std::size_t level) noexcept {
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

  template<class T>
  class FullTreeView : public FullTreeViewBase {
   public:
    constexpr FullTreeView(T* root, std::size_t num_buckets) noexcept:
      FullTreeViewBase(num_buckets),
      root_(root)
    {}

    constexpr std::span<T> nodes() const noexcept {
      return std::span<T>{root_, numNodes()};
    }

    constexpr T& root() const {
      return *root_;
    }

    // Switches to the immediate left subtree.
    constexpr void switchToLeftChild() noexcept {
      root_ += FullTreeViewBase::switchToLeftChild();
    }

    // Switches to the immediate right subtree.
    constexpr void switchToRightChild() noexcept {
      root_ += FullTreeViewBase::switchToRightChild();
    }

   private:
    // Root of the tree.
    T* root_;
  };

  struct FullTreeViewData {
    // Depth of the root of the currently effective tree.
    std::size_t root_level;
    // The number of buckets at the current level.
    std::size_t num_buckets_at_level;
  };

  constexpr FullTreeViewData getEffectiveFullTreeData(std::size_t num_elements,
                                                      std::size_t capacity,
                                                      std::size_t bucket_size) noexcept {
    assert(bucket_size > 1);
    assert(num_elements <= capacity);
    // Number of buckets needed to represent the current elements.
    const std::size_t num_buckets = countBuckets(num_elements, bucket_size);
    // The maximum number of buckets the current tree can represent.
    const std::size_t bucket_capacity = countBuckets(capacity, bucket_size);
    // Level of the currently effective tree.
    const std::size_t root_level = findDeepestNodeForElements(num_buckets, bucket_capacity);
    // The number of buckets at the current level.
    const std::size_t bucket_capacity_at_level = countElementsInLeftmostSubtree(bucket_capacity, root_level);
    return FullTreeViewData{ root_level, bucket_capacity_at_level };
  }

  // Constructs a FullTreeView for the currently effective tree.
  // \param nodes - pointer to the nodes array of the main tree.
  // \param num_elements - the number of currently active elements.
  // \param capacity - the maximum number of elements the tree can represent.
  // \param bucket_size - the number of elements per bucket.
  // \return a FullTreeView for the currently effective tree representing `num_elements` elements.
  template<class T>
  constexpr FullTreeView<T> makeFullTreeView(T* nodes,
                                               std::size_t num_elements,
                                               std::size_t capacity,
                                               std::size_t bucket_size) noexcept {
    const FullTreeViewData tree_data = getEffectiveFullTreeData(num_elements, capacity, bucket_size);
    return FullTreeView<T> {nodes + tree_data.root_level, tree_data.num_buckets_at_level};
  }
}
}
