#pragma once

#include <cumulative_histogram/CumulativeHistogramImpl.h>
#include <cumulative_histogram/FullTreeView.h>

#include <vector>

namespace CumulativeHistogram_NS
{
namespace Detail_NS
{
  class PathEntry : public FullTreeViewBase {
   public:
    constexpr explicit PathEntry(std::size_t root_offset, std::size_t num_buckets) noexcept :
      FullTreeViewBase(num_buckets),
      root_offset_(root_offset)
    {}

    constexpr std::size_t rootOffset() const noexcept { return root_offset_; }

    constexpr void switchToLeftChild() noexcept {
      root_offset_ += FullTreeViewBase::switchToLeftChild();
    }

    constexpr void switchToRightChild() noexcept {
      root_offset_ += FullTreeViewBase::switchToRightChild();
    }

    constexpr void switchToLeftmostChild(std::size_t level) noexcept {
      root_offset_ += FullTreeViewBase::switchToLeftmostChild(level);
    }

    constexpr void switchToRightmostChild(std::size_t level) noexcept {
      root_offset_ += FullTreeViewBase::switchToRightmostChild(level);
    }

    constexpr PathEntry leftChild() const noexcept {
      PathEntry tree = *this;
      tree.switchToLeftChild();
      return tree;
    }

    constexpr PathEntry rightChild() const noexcept {
      PathEntry tree = *this;
      tree.switchToRightChild();
      return tree;
    }

    constexpr PathEntry leftmostChild(std::size_t level) const noexcept {
      PathEntry tree = *this;
      tree.switchToLeftmostChild(level);
      return tree;
    }

    constexpr PathEntry rightmostChild(std::size_t level) const noexcept {
      PathEntry tree = *this;
      tree.switchToRightmostChild(level);
      return tree;
    }

   private:
    // 0-based offset of the root of this tree from the current root of the main tree.
    // By storing the offset rather than a pointer or the absolute index, we ensure that
    // the path remains valid even if reserve() extends the tree.
    std::size_t root_offset_;
  };

  // Stores a compressed path to the last bucket in the tree.
  class CompressedPath {
   public:
    struct Entry {
      PathEntry node;
      std::size_t level; // : std::numeric_limits<std::size_t>::digits;
      // TODO: store instead a single bool in CompressedPath itself, indicating if the last entry
      // is a left subtree or a right subtree. The rest can be deduced.
      bool is_left_subtree; // : 1;
    };

    constexpr explicit CompressedPath() noexcept = default;

    explicit CompressedPath(std::size_t bucket_capacity):
      bucket_capacity_(bucket_capacity),
      num_buckets_(0),
      root_level_(findDeepestNodeForElements(0, bucket_capacity))
    {
      path_.reserve(maxPathLength(bucket_capacity));
    }

    constexpr std::span<const Entry> path() const noexcept { return path_; }

    // Time complexity: O(logN).
    void build(std::size_t num_buckets, std::size_t bucket_capacity) {
      path_.clear();
      path_.reserve(maxPathLength(bucket_capacity));
      bucket_capacity_ = bucket_capacity;
      num_buckets_ = num_buckets;
      // The number of buckets remains the same, but the level of the root may change.
      root_level_ = findDeepestNodeForElements(num_buckets_, bucket_capacity);
      // Construct a path to the last bucket.
      const std::size_t bucket_capacity_at_level = countElementsInLeftmostSubtree(bucket_capacity, root_level_);
      PathEntry tree{ 0, bucket_capacity_at_level };
      // The path is empty if the tree is nodeless.
      if (tree.empty()) {
        return;
      }
      // Otherwise, if the tree has at least 1 node, then the left subtree must be full and the right
      // subtree must not be empty, which means that the first entry will be for a right subtree.
      tree.switchToRightChild();
      bool is_adding_left_subtree = false;
      std::size_t node_level = 1;
      while (!tree.empty()) {
        // Check if the last bucket is in the left subtree.
        const bool last_bucket_is_in_left_subtree = num_buckets_ <= tree.pivot();
        // If we are currently adding an entry for a node which is a leftmost (rightmost)
        // child of some node, and the last bucket is to the left (right) of the current root,
        // then just increase the depth of the node.
        if (last_bucket_is_in_left_subtree == is_adding_left_subtree) {
          ++node_level;
        }
        else {
          path_.push_back(Entry{ .node = tree, .level = node_level, .is_left_subtree = is_adding_left_subtree });
          is_adding_left_subtree = !is_adding_left_subtree;
          node_level = 1;
        }
        // Switch to the relevant subtree.
        if (last_bucket_is_in_left_subtree) {
          tree.switchToLeftChild();
        }
        else {
          tree.switchToRightChild();
        }
      }
      // Add an entry for the last node.
      path_.push_back(Entry{ .node = tree, .level = node_level, .is_left_subtree = is_adding_left_subtree });
    }

    // Update the path according to the new bucket capacity of the tree.
    // Time complexity: O(logN), where N is the new bucket capacity.
    void reserve(std::size_t bucket_capacity) {
      // TODO: optimize the case when the old tree is a subtree of the new tree.
      build(num_buckets_, bucket_capacity);
    }

    // Appends a bucket to the end.
    // Time complexity: O(1).
    void pushBack() {
      // Special case - adding the first bucket. The tree still has 0 nodes
      // after that, so we only need to increase the number of buckets.
      if (num_buckets_ == 0) {
        ++num_buckets_;
        return;
      }

      // OK, we know that there's at least 1 bucket. If there are 2 or more buckets,
      // then the path must not be empty; if there is exactly 1 bucket, then we can treat
      // the *fake* root node as the left subtree of the new root node that will be added after push_back().
      //assert(!path_.empty());
      if (path_.empty() || path_.back().is_left_subtree) {
        // The last entry is the leftmost subtree at level K of some node.
        // If K > 1, then we replace it with an entry for the leftmost subtree at level (K-1) of the same node.
        // Otherwise (if K == 1), we simply remove this entry.
        switchToImmediateParent();
        // Add another entry for the immediate right subtree of the new last entry.
        addEntryForRightChild();
      }
      else {
        // OK, the last entry is the rightmost subtree of some node.
        // Remove the last entry.
        path_.pop_back();
        // The new last entry must be the leftmost subtree at level M of some other node.
        // That node cannot be the root, because that would mean that M == 0, and we don't store 0-level entries in the path.
        // We want to replace this entry with an entry for the leftmost subtree at level (M-1) if M > 1, or remove it altogether
        // if M == 1.
        switchToImmediateParent();
        // If the last entry is a left subtree of some node, append an entry for its immediate right subtree.
        // Otherwise, if the last entry is a right subtree at level L of some node, replace it with an entry
        // for the right subtree at level L+1.
        addEntryForRightChild();
        // If the previously added entry is not a leaf, add an entry for its deepest leftmost subtree.
        if (!path_.back().node.empty()) {
          path_.push_back(makeEntryForDeepestLeftmostSubtree(path_.back().node));
        }
      }
      ++num_buckets_;
      // num_buckets_ >= 2 now, so the path must not be empty.
      assert(!path_.empty());
      // The last node in the path must be a leaf.
      assert(path_.back().node.empty());
    }

    // Removes the last bucket.
    // Time complexity: O(1).
    void popBack() {
      // if ends with a right subtree:
      //   switchToImmediateParent();
      //   if the path is now empty:
      //     increment root level;
      //   else:
      //     addEntryForLeftChild();
      //   // The height of the left subtree may exceed the height of the right subtree by 1,
      //   // so we might need to add one more entry.
      //   addEntryForDeepestRightmostSubtree();
      // else:
      //   remove the last entry;
      //   switch to the immediate parent
      //   add an entry for the left child or update the root if the path is now empty
      //   add an entry for the deepest rightmost subtree.
    }

   private:
    // Time complexity: O(1).
    void switchToImmediateParent() noexcept {
      // Special case: switching to the immediate parent of the root node.
      if (path_.empty()) {
        // switchToImmediateParent() must not be called for the root node if the level of the root is 0 -
        // in this case we must extend the main tree, but we cannot know whether the new bucket capacity should
        // be 2*bucket_capacity_ or (2*bucket_capacity_ - 1).
        assert(root_level_ > 0);
        --root_level_;
        return;
      }
      Entry& last_entry = path_.back();
      if (last_entry.is_left_subtree) {
        if (last_entry.level > 1) {
          // TODO: implement switching to the parent for the only entry in the path.
          assert(path_.size() >= 2);
          // Replace the entry for the leftmost subtree at level M with an entry for the leftmost subtree at level (M-1).
          last_entry.node = path_[path_.size() - 2].node.leftmostChild(last_entry.level - 1);
          --last_entry.level;
        }
        else {
          // Remove the entry for the leftmost subtree at level M == 1.
          path_.pop_back();
        }
      }
      else {
        // TODO: implement.
        assert(false);
      }
    }

    // If the last entry in the path is a left subtree of some node, appends an entry for its immediate right subtree.
    // Otherwise, if the last entry is a right subtree at level L of some node, replaces it with an entry
    // for the right subtree at level L+1.
    // Time complexity: O(1).
    void addEntryForRightChild() noexcept {
      // Special case: adding an entry for the immediate right subtree of the root.
      if (path_.empty()) {
        path_.push_back(Entry{ .node = getRootEntry().rightChild(), .level = 1, .is_left_subtree = false});
        return;
      }
      Entry& last_entry = path_.back();
      // The last entry must not be for a leaf node.
      assert(!last_entry.node.empty());
      if (last_entry.is_left_subtree) {
        path_.push_back(Entry{ .node = last_entry.node.rightChild(), .level = 1, .is_left_subtree = false });
      }
      else {
        last_entry.node.switchToRightChild();
        ++last_entry.level;
      }
    }

    // Constructs an entry for the deepest leftmost subtree of the given PathEntry.
    // The behavior is unspecified if path_entry.empty().
    // \return an entry for the deepest leftmost subtree of path_entry.
    // Time complexity: O(1).
    static constexpr Entry makeEntryForDeepestLeftmostSubtree(const PathEntry& path_entry) noexcept {
      assert(!path_entry.empty());
      const std::size_t level = findDeepestNodeForElements(1, path_entry.numBuckets()); // ceil(log2(num_buckets)).
      return Entry{ .node = path_entry.leftmostChild(level), .level = level, .is_left_subtree = true };
    }

    // Returns the current number of buckets in the tree.
    std::size_t numBuckets() const noexcept {
      return num_buckets_;
      //if (path_.empty()) {
      //  return 0;
      //}
      //const Entry& last_entry = path_.back();
      //return last_entry.node.bucketFirst() + last_entry.node.numBuckets();
    }

    // Root itself is not stored in the path, but we need to use it sometimes.
    PathEntry getRootEntry() const noexcept {
      const std::size_t bucket_capacity_current = countElementsInLeftmostSubtree(bucket_capacity_, root_level_);
      return PathEntry(0, bucket_capacity_current);
    }

    static constexpr std::size_t maxPathLength(std::size_t bucket_capacity) noexcept {
      // Let f(N) be the height of the full tree representing N buckets, i.e.
      // the maximum length of the path from the root (inclusive) to a leaf (inclusive):
      // f(0) = 0    f(4) = 3    f(8) = 4     f(12) = 5    f(16) = 5
      // f(1) = 1    f(5) = 4    f(9) = 5     f(13) = 5    f(17) = 6
      // f(2) = 2    f(6) = 4    f(10) = 5    f(14) = 5    ...
      // f(3) = 3    f(7) = 4    f(11) = 5    f(15) = 5
      // f(2N) = 1 + f(N)
      // f(2N+1) = 1 + f(N+1)
      // I.e. f(N) = 1 + ceil(log2(N)) for N >= 1.
      // We don't store the root in the path, so the maximum number of nodes in the path
      // is f(N)-1 = ceil(log2(N)).
      return bucket_capacity == 0 ? 0 : ceilLog2(bucket_capacity);
    }

    std::vector<Entry> path_;
    // TODO: just store an entry for the root.
    std::size_t bucket_capacity_ = 0;
    std::size_t num_buckets_ = 0;
    std::size_t root_level_ = 0;
  };
}
}
