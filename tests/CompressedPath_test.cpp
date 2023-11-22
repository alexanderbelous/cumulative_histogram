#include <cumulative_histogram/CompressedPath.h>
#include <cumulative_histogram/Math.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <ostream>
#include <string_view>

namespace CumulativeHistogram_NS {
namespace Detail_NS {

static std::ostream& operator<<(std::ostream& stream, const PathEntry& entry) {
  return stream << "{root_offset: " << entry.rootOffset()
                << " bucket_first: " << entry.bucketFirst()
                << " num_buckets: " << entry.numBuckets() << "}";
}

static std::ostream& operator<<(std::ostream& stream, const CompressedPath::Entry& entry) {
  return stream << "{level: " << entry.level
                << " node: " << entry.node << "}";
}

static std::ostream& operator<<(std::ostream& stream, const CompressedPath& path) {
  stream << "{bucket_capacity: " << path.bucketCapacity()
         << " num_buckets: " << path.numBuckets()
         << " root_level: " << path.rootLevel()
         << " path: [";
  const std::span<const CompressedPath::Entry> entries = path.path();
  auto iter = entries.begin();
  if (iter != entries.end()) {
    stream << *iter;
    for (++iter; iter != entries.end(); ++iter) {
      stream << ", " << *iter;
    }
  }
  return stream << "]}";
}

static constexpr bool operator==(const PathEntry& lhs, const PathEntry& rhs) noexcept {
  return lhs.bucketFirst() == rhs.bucketFirst() &&
         lhs.numBuckets() == rhs.numBuckets() &&
         lhs.rootOffset() == rhs.rootOffset();
}

static constexpr bool operator!=(const PathEntry& lhs, const PathEntry& rhs) noexcept {
  return !(lhs == rhs);
}

static constexpr bool operator==(const CompressedPath::Entry& lhs, const CompressedPath::Entry& rhs) noexcept {
  return lhs.level == rhs.level &&
         lhs.node == rhs.node;
}

static constexpr bool operator!=(const CompressedPath::Entry& lhs, const CompressedPath::Entry& rhs) noexcept {
  return !(lhs == rhs);
}

static constexpr bool operator==(const CompressedPath& lhs, const CompressedPath& rhs) noexcept {
  const std::span<const CompressedPath::Entry> entries_lhs = lhs.path();
  const std::span<const CompressedPath::Entry> entries_rhs = rhs.path();
  return lhs.bucketCapacity() == rhs.bucketCapacity() &&
         lhs.numBuckets() == rhs.numBuckets() &&
         lhs.rootLevel() == rhs.rootLevel() &&
         std::equal(entries_lhs.begin(), entries_lhs.end(), entries_rhs.begin(), entries_rhs.end());
}

static constexpr bool operator!=(const CompressedPath& lhs, const CompressedPath& rhs) noexcept {
  return !(lhs == rhs);
}

namespace {

struct CompressedPathData
{
  std::vector<CompressedPath::Entry> path;
  std::size_t bucket_capacity;
  std::size_t num_buckets;
  std::size_t root_level;
};

CompressedPathData getCompressedPathData(const CompressedPath& path)
{
  const std::span<const CompressedPath::Entry> entries = path.path();
  return { .path = std::vector<CompressedPath::Entry>{entries.begin(), entries.end()},
            .bucket_capacity = path.bucketCapacity(),
            .num_buckets = path.numBuckets(),
            .root_level = path.rootLevel() };
}

std::ostream& operator<<(std::ostream& stream, const CompressedPathData& path) {
  stream << "{bucket_capacity: " << path.bucket_capacity
          << " num_buckets: " << path.num_buckets
          << " root_level: " << path.root_level
          << " path: [";
  auto iter = path.path.begin();
  if (iter != path.path.end()) {
    stream << *iter;
    for (++iter; iter != path.path.end(); ++iter) {
      stream << ", " << *iter;
    }
  }
  return stream << "]}";
}

constexpr bool operator==(const CompressedPathData& lhs, const CompressedPathData& rhs) noexcept {
  return lhs.bucket_capacity == rhs.bucket_capacity &&
         lhs.num_buckets == rhs.num_buckets &&
         lhs.root_level == rhs.root_level &&
         lhs.path == rhs.path;
}

constexpr bool operator!=(const CompressedPathData& lhs, const CompressedPathData& rhs) noexcept {
  return !(lhs == rhs);
}

// Similar to FullTreeView, but
//   * skips inactive nodes when switching to subtrees.
//   * stores the absolute index of the root node instead of a pointer.
class TreeViewNumeric {
 public:
  // Expects that 0 < num_buckets <= bucket_capacity.
  constexpr TreeViewNumeric(std::size_t num_buckets,
                            std::size_t bucket_capacity) noexcept :
    TreeViewNumeric(findDeepestNodeForElements(num_buckets, bucket_capacity), num_buckets, bucket_capacity)
  {}

  // Returns true if the tree has no nodes, false otherwise.
  constexpr bool empty() const noexcept {
    // Same as numNodes() == 0
    return bucket_capacity_ <= 1;
  }

  // Returns 0-based index of the root node of the current tree.
  constexpr std::size_t root() const noexcept {
    return root_;
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

  // Switches to the immediate left subtree of the current tree.
  constexpr void switchToLeftChild() noexcept {
    // The left subtree (if it exists) should always be at full capacity.
    ++bucket_capacity_ >>= 1;
    //bucket_capacity_ = (bucket_capacity_ + 1) / 2;  // ceil(capacity_ / 2)
    num_buckets_ = bucket_capacity_;
    root_ += 1;
  }

  // Switches to the effective right subtree from the current root.
  constexpr void switchToRightChild() noexcept {
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
    root_ += (1 + num_nodes_left + level);
  }

  constexpr TreeViewNumeric leftChild() const noexcept {
    TreeViewNumeric tree = *this;
    tree.switchToLeftChild();
    return tree;
  }

  constexpr TreeViewNumeric rightChild() const noexcept {
    TreeViewNumeric tree = *this;
    tree.switchToRightChild();
    return tree;
  }

 private:
  // Expects that 0 < num_buckets <= bucket_capacity.
  constexpr TreeViewNumeric(std::size_t root_level,
                            std::size_t num_buckets,
                            std::size_t bucket_capacity) noexcept :
    root_(root_level),
    bucket_first_(0),
    num_buckets_(num_buckets),
    bucket_capacity_(countElementsInLeftmostSubtree(bucket_capacity, root_level))
  {}

  // 0-based index of the root node.
  std::size_t root_;
  // Index of the first bucket represented by the tree.
  std::size_t bucket_first_;
  // The number of real buckets represented by the tree.
  std::size_t num_buckets_;
  // The maximum number of buckets this tree can represent.
  std::size_t bucket_capacity_;
};

// Checks that the path actually leads to the last bucket.
testing::AssertionResult CheckPath(const CompressedPath& path) {
  PathEntry tree = path.getRootEntry();
  // path[0] is a right subtree, path[1] is a left subtree, and so on.
  bool should_switch_left = false;
  for (const CompressedPath::Entry& entry : path.path()) {
    // The path should end with a leaf.
    if (tree.empty()) {
      return testing::AssertionFailure() << "The path has entries after reaching a leaf (" << tree << "): " << path;
    }
    // Each entry should be a subtree at level K > 0.
    if (entry.level == 0) {
      return testing::AssertionFailure() << "The path has a 0-level entry (" << entry << "): " << path;
    }
    // The depth of the deepest leftmost subtree is ceil(log2(N)),
    // and the depth of the deepest rightmost subtree is floor(log2(N)),
    // where N is the number of buckets in the current tree.
    const std::size_t max_valid_level = should_switch_left ? ceilLog2(tree.numBuckets()) :
                                                             floorLog2(tree.numBuckets());
    // Check that the specified subtree exists.
    if (entry.level > max_valid_level) {
      const std::string_view subtree_type = should_switch_left ? std::string_view("leftmost")
                                                               : std::string_view("rightmost");
      return testing::AssertionFailure() <<
        "The path has an entry for a " << subtree_type << " subtree with an invalid level : " << entry <<
        ". The previous entry led to the subtree " << tree <<
        ", whose deepest " << subtree_type << " child is at level " << max_valid_level <<
        ". Full path: " << path;
    }
    // Switch to the specified subtree.
    if (should_switch_left) {
      tree.switchToLeftmostChild(entry.level);
    }
    else {
      tree.switchToRightmostChild(entry.level);
    }
    // Check that the node stored in the entry is correct.
    if (tree != entry.node) {
      return testing::AssertionFailure() <<
        "Incorrect node in the path entry " << entry <<
        ". Following the path to this entry leads to " << tree <<
        ". Full path: " << path;
    }
    // During the next iteration we should switch to the opposite subtree.
    should_switch_left = !should_switch_left;
  }
  // The path should lead to a leaf node.
  if (!tree.empty()) {
    return testing::AssertionFailure() <<
      "The path leads to a non-leaf node " << tree << ". Full path: " << path;
  }
  // If there is at least 1 bucket, then this leaf node should represent the last bucket.
  if (path.numBuckets() > 0) {
    if ((tree.bucketFirst() != path.numBuckets() - 1) || (tree.numBuckets() != 1)) {
      return testing::AssertionFailure() <<
        "The path " << path << " does not lead to a leaf node representing the last bucket; "
        "instead, it leads to " << tree;
    }
  }
  return testing::AssertionSuccess();
}

TEST(CompressedPath, Build) {
  constexpr std::size_t kMaxBucketCapacity = 20;
  for (std::size_t bucket_capacity = 0; bucket_capacity < kMaxBucketCapacity; ++bucket_capacity) {
    for (std::size_t num_buckets = 0; num_buckets <= bucket_capacity; ++num_buckets) {
      CompressedPath path;
      path.build(num_buckets, bucket_capacity);
      // Validate the path.
      EXPECT_TRUE(CheckPath(path));
    }
  }
}

TEST(CompressedPath, ReserveLessOrSameAsCurrentCapacity) {
  for (std::size_t bucket_capacity = 1; bucket_capacity <= 32; ++bucket_capacity) {
    for (std::size_t num_buckets = 0; num_buckets < bucket_capacity; ++num_buckets) {
      CompressedPath path;
      path.build(num_buckets, bucket_capacity);
      for (std::size_t new_capacity = 0; new_capacity <= bucket_capacity; ++new_capacity) {
        CompressedPath path2;
        path2.build(num_buckets, bucket_capacity);
        EXPECT_EQ(path2, path);
        //CompressedPath path2 = path;
        path2.reserve(new_capacity);
        EXPECT_EQ(path2, path);
      }
    }
  }
}

TEST(CompressedPath, ReserveMoreThanCurrentCapacity) {
  for (std::size_t bucket_capacity = 1; bucket_capacity <= 32; ++bucket_capacity) {
    for (std::size_t num_buckets = 0; num_buckets < bucket_capacity; ++num_buckets) {
      for (std::size_t new_capacity = bucket_capacity + 1; new_capacity <= 64; ++new_capacity) {
        CompressedPath path;
        path.build(num_buckets, bucket_capacity);
        path.reserve(new_capacity);
        EXPECT_EQ(path.bucketCapacity(), new_capacity);
        EXPECT_EQ(path.numBuckets(), num_buckets);
        EXPECT_TRUE(CheckPath(path));
      }
    }
  }
}

TEST(CompressedPath, PushBack) {
  constexpr std::size_t kBucketCapacityMin = 1;
  constexpr std::size_t kBucketCapacityMax = 128;
  for (std::size_t bucket_capacity = kBucketCapacityMin; bucket_capacity <= kBucketCapacityMax; ++bucket_capacity) {
    std::size_t num_buckets = 0;
    CompressedPath path; // { bucket_capacity };
    path.reserve(bucket_capacity);
    for (std::size_t i = 0; i < bucket_capacity; ++i) {
      ++num_buckets;
      path.pushBack();
      EXPECT_EQ(path.bucketCapacity(), bucket_capacity);
      EXPECT_EQ(path.numBuckets(), num_buckets);
      // Check that the path leads to the last bucket.
      EXPECT_TRUE(CheckPath(path));
    }
  }
}

TEST(CompressedPath, PopBack) {
  // The test CompressedPath.PushBack already checks that CompressedPath::pushBack() works
  // correctly, so it's sufficint to check that calling popBack() immediately after pushBack()
  // is a no-op.
  constexpr std::size_t kBucketCapacityMin = 1;
  constexpr std::size_t kBucketCapacityMax = 128;
  for (std::size_t bucket_capacity = kBucketCapacityMin; bucket_capacity <= kBucketCapacityMax; ++bucket_capacity) {
    CompressedPath path;
    path.reserve(bucket_capacity);
    for (std::size_t i = 0; i < bucket_capacity; ++i) {
      // Make a temporary copy of the path.
      const CompressedPathData path_old = getCompressedPathData(path);
      // Add a bucket.
      path.pushBack();
      // Immediately remove the last bucket.
      path.popBack();
      // Check that the path is now in the same state as it was before pushBack().
      const CompressedPathData path_new = getCompressedPathData(path);
      EXPECT_EQ(path_new, path_old);
      // Add a bucket again.
      path.pushBack();
    }
  }
}

TEST(CompressedPath, FindTreeToExtendAfterPushBack) {
  constexpr std::size_t kBucketCapacityMin = 1;
  constexpr std::size_t kBucketCapacityMax = 128;
  for (std::size_t bucket_capacity = kBucketCapacityMin; bucket_capacity <= kBucketCapacityMax; ++bucket_capacity) {
    CompressedPath path;
    path.reserve(bucket_capacity);
    for (std::size_t num_buckets = 0; num_buckets < bucket_capacity;) {
      // Add a bucket.
      path.pushBack();
      ++num_buckets;
      // Find the topmost rightmost subtree at full capacity (time complexity: O(log(num_buckets)).
      TreeViewNumeric tree(num_buckets, bucket_capacity);
      while (!tree.empty() && tree.numBuckets() != tree.bucketCapacity()) {
        tree.switchToRightChild();
      }
      // Traversing the tree this way must lead to some subtree (possibly a leaf) at full capacity.
      EXPECT_EQ(tree.numBuckets(), tree.bucketCapacity());
      // CompressedPath::findTreeToExtendAfterPushBack() must return the same subtree.
      const PathEntry actual = findTreeToExtendAfterPushBack(path);
      EXPECT_EQ(path.rootLevel() + actual.rootOffset(), tree.root());
      EXPECT_EQ(actual.bucketFirst(), tree.bucketFirst());
      EXPECT_EQ(actual.numBuckets(), tree.numBuckets());
    }
  }
}

}
}
}
