#pragma once

#include <cumulative_histogram/FullTreeView.h>
#include <cumulative_histogram/Math.h>

#include <cassert>
#include <vector>

namespace CumulativeHistogram_NS::Detail_NS
{

// Similar to FullTreeView, but stores the offset of the root of the subtree from
// the root of some main tree instead of a pointer to the actual node.
class PathEntry : public FullTreeViewBase
{
public:
  constexpr explicit PathEntry(std::size_t root_offset, std::size_t num_buckets) noexcept :
    FullTreeViewBase(num_buckets),
    root_offset_(root_offset)
  {}

  constexpr std::size_t rootOffset() const noexcept
  {
    return root_offset_;
  }

  constexpr void switchToLeftChild() noexcept
  {
    root_offset_ += FullTreeViewBase::switchToLeftChild();
  }

  constexpr void switchToRightChild() noexcept
  {
    root_offset_ += FullTreeViewBase::switchToRightChild();
  }

  constexpr void switchToLeftmostChild(std::size_t level) noexcept
  {
    root_offset_ += FullTreeViewBase::switchToLeftmostChild(level);
  }

  constexpr void switchToRightmostChild(std::size_t level) noexcept
  {
    root_offset_ += FullTreeViewBase::switchToRightmostChild(level);
  }

  constexpr PathEntry leftChild() const noexcept
  {
    PathEntry tree = *this;
    tree.switchToLeftChild();
    return tree;
  }

  constexpr PathEntry rightChild() const noexcept
  {
    PathEntry tree = *this;
    tree.switchToRightChild();
    return tree;
  }

  constexpr PathEntry leftmostChild(std::size_t level) const noexcept
  {
    PathEntry tree = *this;
    tree.switchToLeftmostChild(level);
    return tree;
  }

  constexpr PathEntry rightmostChild(std::size_t level) const noexcept
  {
    PathEntry tree = *this;
    tree.switchToRightmostChild(level);
    return tree;
  }

private:
  // 0-based offset of the root of this tree from the current root of the main tree.
  // By storing the offset rather than a pointer or the absolute index, we ensure that
  // the path remains valid even if nodes are reallocated.
  std::size_t root_offset_;
};

// Stores a compressed path to the last bucket in the tree.
class CompressedPath
{
public:
  struct Entry
  {
    PathEntry node;
    std::size_t level;
  };

  // Constructs an empty path.
  constexpr explicit CompressedPath() noexcept = default;

  explicit CompressedPath(std::size_t bucket_capacity):
    bucket_capacity_(bucket_capacity),
    num_buckets_(0),
    root_level_(findDeepestNodeForElements(0, bucket_capacity))
  {
    path_.reserve(maxPathLength(bucket_capacity));
  }

  constexpr CompressedPath(const CompressedPath&) = default;

  constexpr CompressedPath(CompressedPath&& other) noexcept;

  constexpr CompressedPath& operator=(const CompressedPath&) = default;

  constexpr CompressedPath& operator=(CompressedPath&& other)
    noexcept(std::is_nothrow_move_assignable_v<std::vector<Entry>>);

  constexpr ~CompressedPath() = default;

  // Returns the compressed path to the last bucket, i.e. the nodes between the root (exclusive)
  // and the leaf (inclusive).
  // Time complexity: O(1).
  constexpr std::span<const Entry> path() const noexcept {
    return path_;
  }

  // Constructs a compressed path to the last bucket for a tree of the specified capacity and size.
  // \param num_buckets - the number of currently active buckets in the tree.
  // \param bucket_capacity - the maximum number of buckets the tree can represent.
  // Time complexity: O(logN).
  inline void build(std::size_t num_buckets, std::size_t bucket_capacity);

  // Update the path according to the new bucket capacity of the tree.
  // \param bucket_capacity - the number of buckets the tree should be able to represent.
  // The function has no effect if this->bucketCapacity() >= bucket_capacity.
  // Otherwise, the path is updated according to the new tree structure. The number of currently
  // active buckets (i.e. the value returned by this->numBuckets()) remains the same.
  // Time complexity: O(logN), where N is the new bucket capacity.
  inline void reserve(std::size_t bucket_capacity);

  // Sets the number of buckets to zero.
  // The path becomes empty after this call; capacity remains unchanged.
  // Time complexity: O(logN).
  // TODO: make it O(1) - there's no need to zero out Entries.
  inline void clear() noexcept;

  // Swaps the contentss of the current path with the given one.
  // Time complexity: O(1).
  inline void swap(CompressedPath& other) noexcept(std::is_nothrow_swappable_v<std::vector<Entry>&>);

  // Modifies the path so that it leads to the bucket following the one that the path currently leads to.
  // Time complexity: O(1).
  inline void pushBack();

  // Modifies the path so that it leads to the bucket preceding the one that the path currently leads to.
  // Time complexity: O(1).
  inline void popBack();

  // \return the maximum number of buckets that the tree can represent.
  // Time complexity: O(1).
  constexpr std::size_t bucketCapacity() const noexcept;

  // Returns the depth of the root of the currently effective tree.
  // Time complexity: O(1).
  constexpr std::size_t rootLevel() const noexcept;

  // Returns the number of currently active buckets in the tree.
  // Time complexity: O(1).
  constexpr std::size_t numBuckets() const noexcept;

  // \return an entry for the root of the currently effective tree.
  // Time complexity: O(1).
  constexpr PathEntry getRootEntry() const noexcept;

  // \return true if the last entry in the path is the leftmost subtree of some node, false otherwise.
  //         Returns true if the path is empty.
  // Time complexity: O(1).
  constexpr bool lastEntryIsLeftSubtree() const noexcept;

  private:
  // Modifies the path so that it leads to the immediate parent of the node that it currently leads to.
  // Time complexity: O(1).
  inline void switchToImmediateParent() noexcept;

  // Modifies the path so that it leads to the immediate left child of the node that it currently leads to.
  // Time complexity: O(1).
  inline void switchToImmediateLeftChild() noexcept;

  // Modifies the path so that it leads to the immediate right child of the node that it currently leads to.
  // Time complexity: O(1).
  inline void switchToImmediateRightChild() noexcept;

  // Modifies the path so that it leads to the deepest rightmost child of the node that it currently leads to.
  // If the node that the path currently leads to is a leaf, the function has no effect.
  // Time complexity: O(1).
  inline void switchToDeepestRightmostChild() noexcept;

  // Constructs an entry for the deepest leftmost subtree of the given PathEntry.
  // The behavior is unspecified if path_entry.empty().
  // \return an entry for the deepest leftmost subtree of path_entry.
  // Time complexity: O(1).
  static constexpr Entry makeEntryForDeepestLeftmostSubtree(const PathEntry& path_entry) noexcept;

  // Compute the maximum length of a path for a tree of the specified capacity.
  // \param bucket_capacity - the maximum number of buckets the tree can represent.
  // \return the maximum number of nodes between the root (exclusive) and a leaf (inclusive) for
  // a tree that can represent up to `bucket_capacity` buckets.
  // Time complexity: O(1).
  static constexpr std::size_t maxPathLength(std::size_t bucket_capacity) noexcept;

  std::vector<Entry> path_;
  // TODO: just store an entry for the root.
  std::size_t bucket_capacity_ = 0;
  std::size_t num_buckets_ = 0;
  std::size_t root_level_ = 0;
};

inline void swap(CompressedPath& lhs, CompressedPath& rhs)
  noexcept(std::is_nothrow_swappable_v<std::vector<CompressedPath::Entry>&>)
{
  lhs.swap(rhs);
}

// Finds the subtree that will need to be extended after adding a bucket to the given tree.
// \param path_to_last_bucket - compressed path to the currently last bucket in the tree.
// \return a PathEntry to the subtree that will need to be extended after adding one more bucket to the
//         tree described by path_to_last_bucket, i.e. the largest subtree at full capacity that contains
//         the last bucket. Note that this is always the left subtree of some other tree.
// The behavior is undefined if the tree currently has 0 buckets (in that case no nodes will need to
// be constructed after adding the first bucket).
// Time complexity: O(1).
constexpr PathEntry findTreeToExtendAfterPushBack(const CompressedPath& path_to_last_bucket) noexcept
{
  assert(path_to_last_bucket.numBuckets() > 0);
  const std::span<const CompressedPath::Entry> path = path_to_last_bucket.path();
  // 1. Edge case - if there is only 1 bucket, then the path is empty (because the tree has 0 nodes),
  //    and the tree that will need to be extended is the only leaf.
  // 2. If there is exactly 1 entry in the path, then it must be the rightmost subtree (at some level K) of
  //    the root of the currently effective tree. In this case we should extend the currently effective tree.
  if (path.size() < 2) {
    return path_to_last_bucket.getRootEntry();
  }
  // Otherwise, if the last entry is a left subtree of some node, then this leaf is the subtree that will
  // need to be extended.
  if (path_to_last_bucket.lastEntryIsLeftSubtree()) {
    return path.back().node;
  }
  // Otherwise, the last entry is the rightmost subtree (at level K) of some node, which means that the tree
  // that will need to be extended is that parent node.
  return path[path.size() - 2].node;
}

constexpr CompressedPath::CompressedPath(CompressedPath&& other) noexcept :
  path_(std::move(other.path_)),
  bucket_capacity_(std::exchange(other.bucket_capacity_, static_cast<std::size_t>(0))),
  num_buckets_(std::exchange(other.num_buckets_, static_cast<std::size_t>(0))),
  root_level_(std::exchange(other.root_level_, static_cast<std::size_t>(0)))
{}

constexpr CompressedPath& CompressedPath::operator=(CompressedPath&& other)
  noexcept(std::is_nothrow_move_assignable_v<std::vector<Entry>>)
{
  path_ = std::move(other.path_);
  bucket_capacity_ = std::exchange(other.bucket_capacity_, static_cast<std::size_t>(0));
  num_buckets_ = std::exchange(other.num_buckets_, static_cast<std::size_t>(0));
  root_level_ = std::exchange(other.root_level_, static_cast<std::size_t>(0));
  return *this;
}

void CompressedPath::build(std::size_t num_buckets, std::size_t bucket_capacity)
{
  path_.clear();
  path_.reserve(maxPathLength(bucket_capacity));
  bucket_capacity_ = bucket_capacity;
  num_buckets_ = num_buckets;
  root_level_ = findDeepestNodeForElements(num_buckets_, bucket_capacity);
  // Construct a path to the last bucket.
  const std::size_t bucket_capacity_at_level = countElementsInLeftmostSubtree(bucket_capacity, root_level_);
  PathEntry tree{ 0, bucket_capacity_at_level };
  // The path is empty if the tree is nodeless.
  if (tree.empty())
  {
    return;
  }
  // Otherwise, if the tree has at least 1 node, then the left subtree must be full and the right
  // subtree must not be empty, which means that the first entry will be for a right subtree.
  tree.switchToRightChild();
  bool is_adding_left_subtree = false;
  std::size_t node_level = 1;
  while (!tree.empty())
  {
    // Check if the last bucket is in the left subtree.
    const bool last_bucket_is_in_left_subtree = num_buckets_ <= tree.pivot();
    // If we are currently adding an entry for a node which is a leftmost (rightmost)
    // child of some node, and the last bucket is to the left (right) of the current root,
    // then just increase the depth of the node.
    if (last_bucket_is_in_left_subtree == is_adding_left_subtree)
    {
      ++node_level;
    }
    else
    {
      path_.push_back(Entry{ .node = tree, .level = node_level });
      is_adding_left_subtree = !is_adding_left_subtree;
      node_level = 1;
    }
    // Switch to the relevant subtree.
    if (last_bucket_is_in_left_subtree)
    {
      tree.switchToLeftChild();
    }
    else
    {
      tree.switchToRightChild();
    }
  }
  // Add an entry for the last node.
  path_.push_back(Entry{ .node = tree, .level = node_level });
}

void CompressedPath::reserve(std::size_t bucket_capacity)
{
  // Do nothing if the current capacity is greater or equal to the input one.
  if (bucket_capacity_ >= bucket_capacity)
  {
    return;
  }
  // Check if the currently effective tree is a subtree of the new tree.
  // Note that this doesn't necessarily mean that the current full tree is a subtree of the new tree:
  // for example, if the current capacity is 15, and the number of currently active buckets is 8 (i.e.
  // all the active buckets are in the left subtree, which is the currently effective tree), then the
  // currently effective tree is a subtree of the tree representing 32 buckets, even though no leftmost
  // subtree of that tree has capacity equal to 15.
  const std::size_t bucket_capacity_current = countElementsInLeftmostSubtree(bucket_capacity_, root_level_);
  const std::size_t level = findLeftmostSubtreeWithExactCapacity(bucket_capacity_current, bucket_capacity);
  if (level != static_cast<std::size_t>(-1))
  {
    // Allocate extra memory in case the longest path in the new tree is longer that
    // the longest path in the old tree.
    path_.reserve(maxPathLength(bucket_capacity));
    // Update the capacity.
    bucket_capacity_ = bucket_capacity;
    // Update the root level.
    root_level_ = findDeepestNodeForElements(num_buckets_, bucket_capacity);
    return;
  }
  // Otherwise, just rebuild the path.
  build(num_buckets_, bucket_capacity);
}

void CompressedPath::clear() noexcept
{
  path_.clear();
  num_buckets_ = 0;
  root_level_ = findDeepestNodeForElements(0, bucket_capacity_);
}

void CompressedPath::swap(CompressedPath& other) noexcept(std::is_nothrow_swappable_v<std::vector<Entry>&>)
{
  path_.swap(other.path_);
  std::swap(bucket_capacity_, other.bucket_capacity_);
  std::swap(num_buckets_, other.num_buckets_);
  std::swap(root_level_, other.root_level_);
}

void CompressedPath::pushBack()
{
  // Special case - adding the first bucket. The tree still has 0 nodes
  // after that, so we only need to increase the number of buckets.
  if (num_buckets_ == 0)
  {
    ++num_buckets_;
    return;
  }

  // OK, we know that there's at least 1 bucket. If there are 2 or more buckets,
  // then the path must not be empty; if there is exactly 1 bucket, then we can treat
  // the *fake* root node as the left subtree of the new root node that will be added after push_back().
  if (path_.empty() || lastEntryIsLeftSubtree())
  {
    // The last entry is the leftmost subtree at level K of some node.
    // If K > 1, then we replace it with an entry for the leftmost subtree at level (K-1) of the same node.
    // Otherwise (if K == 1), we simply remove this entry.
    switchToImmediateParent();
    // Add another entry for the immediate right subtree of the new last entry.
    switchToImmediateRightChild();
  }
  else
  {
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
    switchToImmediateRightChild();
    // If the previously added entry is not a leaf, add an entry for its deepest leftmost subtree.
    if (!path_.back().node.empty())
    {
      path_.push_back(makeEntryForDeepestLeftmostSubtree(path_.back().node));
    }
  }
  ++num_buckets_;
  // num_buckets_ >= 2 now, so the path must not be empty.
  assert(!path_.empty());
  // The last node in the path must be a leaf.
  assert(path_.back().node.empty());
}

void CompressedPath::popBack()
{
  assert(num_buckets_ != 0);
  // The path can be empty if the tree currently stores exactly 1 bucket.
  if (!path_.empty())
  {
    // If the last entry is a leftmost subtree of some node, remove that entry.
    if (lastEntryIsLeftSubtree())
    {
      path_.pop_back();
    }
    // Now the path must end with an entry for the rightmost subtree of some node.
    // It cannot be empty because even the entry for the leftmost subtree that we might've removed
    // earlier could not be the only entry - the path must always start with an entry for the rightmost
    // subtree.
    // Switch to the immediate parent of the node that the path currently leads to.
    switchToImmediateParent();
    // If the path is now empty, update the root so that it refers to its immediate left child.
    // Otherwise, switch to the immediate left child of the last entry.
    switchToImmediateLeftChild();
    // Now either the path is empty, or it ends with an entry for the leftmost subtree of some node.
    // If the new last entry is not a leaf, add an entry for its deepest rightmost subtree.
    switchToDeepestRightmostChild();
  }
  --num_buckets_;
}

constexpr std::size_t CompressedPath::bucketCapacity() const noexcept
{
  return bucket_capacity_;
}

constexpr std::size_t CompressedPath::rootLevel() const noexcept
{
  return root_level_;
}

constexpr std::size_t CompressedPath::numBuckets() const noexcept
{
  return num_buckets_;
  //if (path_.empty())
  //{
  //  return 0;
  //}
  //const Entry& last_entry = path_.back();
  //return last_entry.node.bucketFirst() + last_entry.node.numBuckets();
}

constexpr bool CompressedPath::lastEntryIsLeftSubtree() const noexcept
{
  // 1. By definition, path_[0] (if it exists) is always the right subtree of the currently
  // effective tree - otherwise, the last bucket would be in the left subtree, which means that the right
  // subtree is empty, which means that left subtree should be the effective tree.
  // 2. path_[1] (if it exists) is always the left subtree of the path_[0] .
  // 3. path_[2] (if it exists) is always the right subtree of the path_[1].
  // 4. And so on. path_[i] (if it exits) is the left subtree of path_[i-1] if i is odd,
  //    otherwise it's the right subtree.
  // 5. Therefore, path_[path_.size() - 1] is the left subtree if path.size() is even, otherwise it's
  //    the right subtree.
  // 6. It doesn't really matter what the function returns if the path is empty, but let's return true -
  //    if the path is empty, then there is at most 1 bucket, which is the left child of the root of the
  //    currently effective tree.
  return path_.size() % 2 == 0;
}

void CompressedPath::switchToImmediateParent() noexcept
{
  // Special case: switching to the immediate parent of the root node.
  if (path_.empty())
  {
    // switchToImmediateParent() must not be called for the root node if the level of the root is 0 -
    // in this case we must extend the main tree, but we cannot know whether the new bucket capacity should
    // be 2*bucket_capacity_ or (2*bucket_capacity_ - 1).
    assert(root_level_ > 0);
    --root_level_;
    return;
  }
  Entry& last_entry = path_.back();
  if (last_entry.level > 1)
  {
    // Construct an entry for the immediate parent.
    // Initialize with the previous entry.
    PathEntry new_entry = (path_.size() >= 2) ? path_[path_.size() - 2].node : getRootEntry();
    // Switch to the leftmost (rightmost) subtree at level M-1.
    if (lastEntryIsLeftSubtree())
    {
      new_entry.switchToLeftmostChild(last_entry.level - 1);
    }
    else
    {
      new_entry.switchToRightmostChild(last_entry.level - 1);
    }
    last_entry.node = new_entry;
    --last_entry.level;
  }
  else
  {
    // Just remove the entry for the leftmost/rightmost subtree at level M == 1.
    path_.pop_back();
  }
}

void CompressedPath::switchToImmediateLeftChild() noexcept
{
  // Special case: switching to the immediate left subtree of the root.
  // In this case instead of adding an entry we just update the root.
  if (path_.empty())
  {
    ++root_level_;
    return;
  }
  Entry& last_entry = path_.back();
  // The last entry must not be for a leaf node.
  assert(!last_entry.node.empty());
  if (lastEntryIsLeftSubtree())
  {
    last_entry.node.switchToLeftChild();
    ++last_entry.level;
  }
  else
  {
    path_.push_back(Entry{ .node = last_entry.node.leftChild(), .level = 1 });
  }
}

void CompressedPath::switchToImmediateRightChild() noexcept
{
  // Special case: adding an entry for the immediate right subtree of the root.
  if (path_.empty())
  {
    path_.push_back(Entry{ .node = getRootEntry().rightChild(), .level = 1 });
    return;
  }
  Entry& last_entry = path_.back();
  // The last entry must not be for a leaf node.
  assert(!last_entry.node.empty());
  if (lastEntryIsLeftSubtree())
  {
    path_.push_back(Entry{ .node = last_entry.node.rightChild(), .level = 1 });
  }
  else
  {
    last_entry.node.switchToRightChild();
    ++last_entry.level;
  }
}

void CompressedPath::switchToDeepestRightmostChild() noexcept
{
  if (path_.empty())
  {
    PathEntry entry = getRootEntry();
    if (entry.empty())
    {
      return;
    }
    // The rightmost subtree at level K has floor(N / 2^K) buckets, where N is the number of buckets in the
    // current tree. Therefore, the maximum valid level is such Kmax that floor(N / 2^Kmax) == 1.
    // I.e. Kmax = floor(log2(N)): this way, N >= 2^Kmax, but also N < 2^(Kmax+1).
    const std::size_t level = floorLog2(entry.numBuckets());
    entry.switchToRightmostChild(level);
    path_.push_back(Entry{ .node = entry, .level = level });
    return;
  }
  Entry& last_entry = path_.back();
  if (last_entry.node.empty())
  {
    return;
  }
  const std::size_t level = floorLog2(last_entry.node.numBuckets());
  if (lastEntryIsLeftSubtree())
  {
    path_.push_back(Entry{ .node = last_entry.node.rightmostChild(level), .level = level });
  }
  else
  {
    last_entry.node.switchToRightmostChild(level);
    last_entry.level += level;
  }
}

constexpr
CompressedPath::Entry
CompressedPath::makeEntryForDeepestLeftmostSubtree(const PathEntry& path_entry) noexcept
{
  assert(!path_entry.empty());
  // The leftmost subtree at level K has ceil(N / 2^K) buckets, where N is the number of buckets in the
  // current tree. The maximum valid level is the smallest Kmax such that ceil(N / 2^Kmax) == 1.
  // I.e. Kmax = ceil(log2(N)).
  const std::size_t level = ceilLog2(path_entry.numBuckets());
  return Entry{ .node = path_entry.leftmostChild(level), .level = level };
}

constexpr PathEntry CompressedPath::getRootEntry() const noexcept
{
  const std::size_t bucket_capacity_current = countElementsInLeftmostSubtree(bucket_capacity_, root_level_);
  return PathEntry(0, bucket_capacity_current);
}

constexpr std::size_t CompressedPath::maxPathLength(std::size_t bucket_capacity) noexcept
{
  // Let f(N) be the height of the full tree representing N buckets, i.e.
  // the maximum length of the path from the root (inclusive) to a leaf (inclusive):
  // f(0) = 0
  // f(1) = 1
  // ...
  // f(2N) = 1 + f(N)
  // f(2N+1) = 1 + f(N+1)
  // I.e. f(N) = 1 + ceil(log2(N)) for N >= 1.
  // We don't store the root in the path, so the maximum number of nodes in the path
  // is f(N)-1 = ceil(log2(N)).
  return bucket_capacity == 0 ? 0 : ceilLog2(bucket_capacity);
}

}  // namespace CumulativeHistogram_NS::Detail_NS
