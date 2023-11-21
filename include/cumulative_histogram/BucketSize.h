#pragma once

#include <cstddef>
#include <type_traits>

namespace CumulativeHistogram_NS
{
namespace Detail_NS
{

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
//
// The time complexities of the operations of CumulativeHistogram with respect to both N and BucketSize:
// +--------------+--------------------------------------+
// | increment()  | O(log(N/BucketSize))                 |
// +--------------+--------------------------------------+
// | prefixSum()  | O(log(N/BucketSize)) + O(BucketSize) |
// +--------------+--------------------------------------+
// | lowerBound() | O(log(N/BucketSize)) + O(BucketSize) |
// +--------------+--------------------------------------+
// | pushBack()   | TODO                                 |
// +--------------+--------------------------------------+
// | popBack()    | O(1)                                 |
// +--------------+--------------------------------------+
template<class T, class Enable = void>
class DefaultBucketSize : public std::integral_constant<std::size_t, 2> {};

// Partial specialization for 8-bit integer types.
template<class T>
class DefaultBucketSize<T, std::enable_if_t<std::is_integral_v<T> && sizeof(T) == 1>> :
  public std::integral_constant<std::size_t, 128> {};

// Partial specialization for 16-bit integer types.
template<class T>
class DefaultBucketSize<T, std::enable_if_t<std::is_integral_v<T> && sizeof(T) == 2>> :
  public std::integral_constant<std::size_t, 128> {};

// Partial specialization for 32-bit integer types.
template<class T>
class DefaultBucketSize<T, std::enable_if_t<std::is_integral_v<T> && sizeof(T) == 4>> :
  public std::integral_constant<std::size_t, 128> {};

// Partial specialization for 64-bit integer types.
template<class T>
class DefaultBucketSize<T, std::enable_if_t<std::is_integral_v<T> && sizeof(T) == 8>> :
  public std::integral_constant<std::size_t, 128> {};

// Partial specialization for floating-point types (float, double, long double).
template<class T>
class DefaultBucketSize<T, std::enable_if_t<std::is_floating_point_v<T>>> :
  public std::integral_constant<std::size_t, 16> {};

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
class BucketSize : public Detail_NS::DefaultBucketSize<T> {};

}  // namespace CumulativeHistogram_NS
