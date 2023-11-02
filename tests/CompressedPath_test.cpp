#include <cumulative_histogram/CompressedPath.h>

#include <gtest/gtest.h>

namespace CumulativeHistogram_NS {
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
        EXPECT_EQ(tree.bucketFirst(), entry.node.bucketFirst());
        EXPECT_EQ(tree.numBuckets(), entry.node.numBuckets());
        EXPECT_EQ(tree.rootOffset(), entry.node.rootOffset());
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
      EXPECT_EQ(tree.bucketFirst(), entry.node.bucketFirst());
      EXPECT_EQ(tree.numBuckets(), entry.node.numBuckets());
      EXPECT_EQ(tree.rootOffset(), entry.node.rootOffset());
    }
    // The path should end with a leaf node.
    EXPECT_TRUE(tree.empty());
  }
}

}
}