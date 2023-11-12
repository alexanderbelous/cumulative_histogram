#pragma once

#include <type_traits>

namespace CumulativeHistogram_NS {
namespace Detail_NS {

// DefaultBucketSize provides the optimal bucket sizes for built-in arithmetic types. Users should not
// provide specializations for this template - instead, they should define specializations for
// ::CumulativeHistogram_NS::BucketSize.
//
// Storing 2 elements per bucket is a reasonable default because it minimizes the number of additions when
// computing a prefix sum. However, CumulativeHistogram::prefixSum() also needs to traverse the auxiliary
// tree, which takes O(log(N/BucketSize)) time, where N is the current number of elements. The constant
// factor in this time complexity is significant enough to justify using larger buckets for types T for which
// addition is fast. Moreover, using larger buckets can improve performance for built-in arithmetic types by
// enabling vectorization.
template<class T, class Enable = void>
class DefaultBucketSize : public std::integral_constant<std::size_t, 2> {};

// Partial specialization for 32-bit integer types: bucket size is 128.
template<class T>
class DefaultBucketSize<T, std::enable_if_t<std::is_integral_v<T> && sizeof(T) == 4>> :
  public std::integral_constant<std::size_t, 128> {};

}  // namespace Detail_NS

// Type trait that provides the bucket size to use in the class CumulativeHistogram<T>.
// Uses are allowed to provide specializations (both full and partial) for this template, e.g.,
//
//   // Full specialization for int - overrides the default bucket size with 32.
//   template<>
//   class BucketSize<int> : public std::integral_constant<std::size_t, 32> {};
//
//   // Partial specialization for all floating-point types - overrides the default bucket size with 16.
//   template<class T>
//   class BucketSize<T, std::enable_if_t<std::is_floating_point_v<T>> :
//     public std::integral_constant<std::size_t, 16> {};
//
template<class T, class Enable = void>
class BucketSize : public ::CumulativeHistogram_NS::Detail_NS::DefaultBucketSize<T> {};
}
