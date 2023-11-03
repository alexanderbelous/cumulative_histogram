#pragma once

#include <cumulative_histogram/CumulativeHistogramImpl.h>

#include <span>
#include <type_traits>

namespace CumulativeHistogram_NS
{
namespace Detail_NS
{
  // Non-template base for TreeView.
  class TreeViewBase {
   public:
    constexpr TreeViewBase(std::size_t num_buckets, std::size_t bucket_capacity) noexcept:
      bucket_first_(0),
      num_buckets_(num_buckets),
      bucket_capacity_(bucket_capacity)
    {}

    // Returns true if the tree has no nodes, false otherwise.
    constexpr bool empty() const noexcept {
      // Same as numNodes() == 0
      return bucket_capacity_ <= 1;
    }

    constexpr std::size_t numNodes() const noexcept {
      return countNodesInBucketizedTree(bucket_capacity_);
    }

    constexpr std::size_t bucketFirst() const noexcept {
      return bucket_first_;
    }

    constexpr std::size_t numBuckets() const noexcept {
      return num_buckets_;
    }

    constexpr std::size_t bucketCapacity() const noexcept {
      return bucket_capacity_;
    }

    // Returns 0-based index of the last bucket (inclusive) of the left subtree.
    constexpr std::size_t pivot() const noexcept {
      return bucket_first_ + (bucket_capacity_ - 1) / 2;
    }

   protected:
    // Returns the offset of the root of the left subtree from the current root.
    constexpr std::size_t switchToLeftChild() noexcept {
      // The left subtree (if it exists) should always be at full capacity.
      ++bucket_capacity_ >>= 1;
      //bucket_capacity_ = (bucket_capacity_ + 1) / 2;  // ceil(capacity_ / 2)
      num_buckets_ = bucket_capacity_;
      return 1;
    }

    // Returns the offset of the root of the effective right subtree from the current root.
    constexpr std::size_t switchToRightChild() noexcept {
      const std::size_t bucket_capacity_left = (bucket_capacity_ + 1) / 2;  // ceil(capacity_ / 2)
      const std::size_t bucket_capacity_right = bucket_capacity_ / 2;       // floor(capacity_ / 2)
      const std::size_t num_nodes_left = countNodesInBucketizedTree(bucket_capacity_left);

      bucket_first_ += bucket_capacity_left;
      num_buckets_ -= bucket_capacity_left;
      // Find the deepest leftmost subtree of the immediate right subtree that represents all
      // real elements of the right subtree.
      const std::size_t level = findDeepestNodeForElements(num_buckets_, bucket_capacity_right);
      bucket_capacity_ = countElementsInLeftmostSubtree(bucket_capacity_right, level);
      // Skip the 0th node because it's the root.
      // Skip the next `num_nodes_left` because they belong to the left subtree.
      // Skip the next `level` nodes because those are nodes between the root of our
      //      "effective" right subtree and the root of the current tree.
      // TODO: this is the same as bucket_capacity_left + level
      return 1 + num_nodes_left + level;
    }

   private:
    // Index of the first bucket represented by the tree.
    std::size_t bucket_first_;
    // The number of real buckets represented by the tree.
    std::size_t num_buckets_;
    // The maximum number of buckets this tree can represent.
    std::size_t bucket_capacity_;
  };

  // High-level API for interacing with the implicit tree data structure.
  template<class T>
  class TreeView : public TreeViewBase {
  public:
    // Expects that 0 < num_buckets <= bucket_capacity.
    constexpr TreeView(std::span<T> nodes,
                       std::size_t num_buckets,
                       std::size_t bucket_capacity) noexcept :
      TreeViewBase(num_buckets, bucket_capacity),
      root_(nodes.data())
    {}

    // Converting constructor for an immutable TreeView from a mutable TreeView.
    template<class Enable = std::enable_if_t<std::is_const_v<T>>>
    constexpr TreeView(const TreeView<std::remove_const_t<T>>& tree) noexcept :
      TreeViewBase(tree),
      root_(tree.nodes().data())
    {}

    constexpr std::span<T> nodes() const noexcept {
      return std::span<T>{root_, numNodes()};
    }

    constexpr T& root() const {
      return *root_;
    }

    constexpr void switchToLeftChild() noexcept {
      root_ += TreeViewBase::switchToLeftChild();
    }

    constexpr TreeView leftChild() const noexcept {
      TreeView tree = *this;
      tree.switchToLeftChild();
      return tree;
    }

    constexpr void switchToRightChild() noexcept {
      root_ += TreeViewBase::switchToRightChild();
    }

    constexpr TreeView rightChild() const noexcept {
      TreeView tree = *this;
      tree.switchToRightChild();
      return tree;
    }

  private:
    T* root_;
  };

  struct TreeViewData {
    // The number of currently active buckets.
    std::size_t num_buckets;
    // Level of the currently effective tree.
    std::size_t root_level;
    // The number of buckets at the current level.
    std::size_t bucket_capacity_at_level;
    // The number of nodes at the current level.
    std::size_t num_nodes_at_level;
  };

  constexpr TreeViewData getEffectiveTreeData(std::size_t num_elements,
                                              std::size_t capacity,
                                              std::size_t bucket_size) noexcept {
    assert(bucket_size > 1);
    assert(num_elements <= capacity);

    // The maximum number of buckets the current tree can represent.
    const std::size_t bucket_capacity = countBuckets(capacity, bucket_size);
    // Number of buckets needed to represent the current elements.
    const std::size_t num_buckets = countBuckets(num_elements, bucket_size);
    // Level of the currently effective tree.
    const std::size_t root_level = findDeepestNodeForElements(num_buckets, bucket_capacity);
    // The number of buckets at the current level.
    const std::size_t bucket_capacity_at_level = countElementsInLeftmostSubtree(bucket_capacity, root_level);
    // The number of nodes at the current level.
    const std::size_t num_nodes_at_level = countNodesInBucketizedTree(bucket_capacity_at_level);
    return TreeViewData{
      .num_buckets = num_buckets,
      .root_level = root_level,
      .bucket_capacity_at_level = bucket_capacity_at_level,
      .num_nodes_at_level = num_nodes_at_level
    };
  }
}
}
