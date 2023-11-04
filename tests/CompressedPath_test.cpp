#include <cumulative_histogram/CompressedPath.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <ostream>

namespace CumulativeHistogram_NS {
namespace Detail_NS {

static std::ostream& operator<<(std::ostream& stream, const PathEntry& entry) {
  return stream << "{root_offset: " << entry.rootOffset()
                << " bucket_first: " << entry.bucketFirst()
                << " num_buckets: " << entry.numBuckets() << "}";
}

static std::ostream& operator<<(std::ostream& stream, const CompressedPath::Entry& entry) {
  return stream << "{is_left_subtree: " << (entry.is_left_subtree ? "true" : "false")
                << " level: " << entry.level
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
  return lhs.is_left_subtree == rhs.is_left_subtree &&
         lhs.level == rhs.level &&
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

TEST(CompressedPath, Build) {
  using ::CumulativeHistogram_NS::Detail_NS::CompressedPath;
  using ::CumulativeHistogram_NS::Detail_NS::PathEntry;
  using ::CumulativeHistogram_NS::Detail_NS::countElementsInLeftmostSubtree;
  using ::CumulativeHistogram_NS::Detail_NS::findDeepestNodeForElements;

  constexpr std::size_t kMaxBucketCapacity = 20;

  for (std::size_t bucket_capacity = 0; bucket_capacity < kMaxBucketCapacity; ++bucket_capacity) {
    for (std::size_t num_buckets = 0; num_buckets <= bucket_capacity; ++num_buckets) {
      CompressedPath path;
      path.build(num_buckets, bucket_capacity);
      // Validate the path.
      const std::size_t root_level = findDeepestNodeForElements(num_buckets, bucket_capacity);
      const std::size_t bucket_capacity_at_level = countElementsInLeftmostSubtree(bucket_capacity, root_level);
      PathEntry tree{ 0, bucket_capacity_at_level };
      for (const CompressedPath::Entry& entry : path.path()) {
        EXPECT_FALSE(tree.empty());
        if (entry.is_left_subtree) {
          tree.switchToLeftmostChild(entry.level);
        }
        else {
          tree.switchToRightmostChild(entry.level);
        }
        // Check the node.
        EXPECT_EQ(tree, entry.node);
      }
      // The path should end with a leaf node.
      EXPECT_TRUE(tree.empty());
      // If the path is not empty, then the last node should represent the last bucket.
      if (!path.path().empty()) {
        EXPECT_EQ(path.path().back().node.bucketFirst() + 1, num_buckets);
      }
    }
  }
}

TEST(CompressedPath, PushBack) {
  using ::CumulativeHistogram_NS::Detail_NS::CompressedPath;
  using ::CumulativeHistogram_NS::Detail_NS::PathEntry;
  using ::CumulativeHistogram_NS::Detail_NS::countElementsInLeftmostSubtree;
  using ::CumulativeHistogram_NS::Detail_NS::findDeepestNodeForElements;

  const std::size_t bucket_capacity = 8;
  std::size_t num_buckets = 0;
  CompressedPath path{ bucket_capacity };

  for (std::size_t i = 0; i < bucket_capacity; ++i) {
    ++num_buckets;
    path.pushBack();

    // Check the path
    const std::size_t root_level = findDeepestNodeForElements(num_buckets, bucket_capacity);
    const std::size_t bucket_capacity_at_level = countElementsInLeftmostSubtree(bucket_capacity, root_level);
    PathEntry tree{ 0, bucket_capacity_at_level };
    for (const CompressedPath::Entry& entry : path.path()) {
      EXPECT_FALSE(tree.empty());
      if (entry.is_left_subtree) {
        tree.switchToLeftmostChild(entry.level);
      }
      else {
        tree.switchToRightmostChild(entry.level);
      }
      // Check the node.
      EXPECT_EQ(tree, entry.node);
    }
    // The path should end with a leaf node.
    EXPECT_TRUE(tree.empty());
  }
}

TEST(CompressedPath, PopBack) {
  // The test CompressedPath.PushBack already checks that CompressedPath::pushBack() works
  // correctly, so it's sufficint to check that calling popBack() immediately after pushBack()
  // is a no-op.
  using ::CumulativeHistogram_NS::Detail_NS::CompressedPath;
  constexpr std::size_t kBucketCapacityMin = 1;
  constexpr std::size_t kBucketCapacityMax = 128;
  for (std::size_t bucket_capacity = kBucketCapacityMin; bucket_capacity <= kBucketCapacityMax; ++bucket_capacity) {
    CompressedPath path{ bucket_capacity };
    for (std::size_t i = 0; i < bucket_capacity; ++i) {
      // Make a temporary copy of the path.
      const CompressedPath path_was = path;
      // Add a bucket.
      path.pushBack();
      // Immediately remove the last bucket.
      path.popBack();
      // Check that the path is now in the same state as it was before pushBack().
      EXPECT_EQ(path, path_was);
      // Add a bucket again.
      path.pushBack();
    }
  }
}

}
}
}