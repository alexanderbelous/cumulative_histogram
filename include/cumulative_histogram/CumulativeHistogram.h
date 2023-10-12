#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace CumulativeHistogram_NS {

// Forward declarations.
namespace Detail_NS {
  template<class T>
  class TreeView;

  template<class T>
  class TreeViewSimple;
}

// Defines a named requirement for an additive type.
//
// This is similar to an additive group in mathematics, except that this concept
// doesn't require every element to have an inverse element (e.g., arbitrary-precision
// unsigned integers satisfy this requirement, even though they don't have negative
// counterparts).
template<typename T>
concept Additive =
  // T should be a semiregular type
  // (i.e. T must be both copyable and default constructible).
  std::semiregular<T> &&
  // Given an rvalue `lhs` of type `T&&` an lvalue `rhs` of type `const T&`,
  // the expression `std::move(lhs) + rhs` must be convertible to `T`.
  requires(T&& lhs, const T& rhs) { { std::move(lhs) + rhs } -> std::convertible_to<T>; } &&
  // Given an lvalue `lhs` of type `const T&` an lvalue `rhs` of type `const T&`,
  // the expression `lhs + rhs` must be convertible to `T`.
  requires(const T& lhs, const T& rhs) { { lhs + rhs } -> std::convertible_to<T>; } &&
  // Given an lvalue `lhs` of type `T&` an lvalue `rhs` of type `const T&`,
  // the expression `lhs += rhs` must be valid.
  requires(T& lhs, const T& rhs) { lhs += rhs; };

// A class for efficient computation of prefix sums for a dynamic array of elements.
//
// The template type parameter T must satisfy the `Additive` concept above. There are additional
// requirements that cannot be expressed via C++ concepts:
// 1) Addition must be commutative, i.e. `a + b == b + a` for any a and b.
// 2) Addition must be associative, i.e. `(a + b) + c == a + (b + c)` for any a, b, c.
// 3) A default-constructed value-initialized object must have the same meaning as the identity element in
//    additive groups ("zero"): `T{} + x == x` for any x.
//
// * Built-in arithmetic types satisfy these requirements.
//   * Note that signed integer overflow is undefined behavior.
//   * Floating-point addition is not always associative due to rounding errors. Hence, computing a prefix
//     sum for a sequence of floating-point values may produce different results depending on the order of
//     operands. You can still use CumulativeHistogram with floating-point types, but note that it doesn't
//     attempt to minimize rounding errors. If you need better precision guarantees, use algorithms like
//     Kahan summation or pairwise summation.
// * std::complex and std::chrono::duration satisfy these requirements (except that they have the same issue
//   with rounding errors when using floating-point versions).
// * User-defined classes for arbitrary-precision integers, N-dimensional vectors, quaternions, etc satisfy
//   these requirements (as long as they overload operator+ and operator+=).
// * std::string does NOT satisfy these requirements because string concatenation is not commutative.
//
// TODO: add a second parameter SumType, which defaults to T.
//       In some scenarios it's reasonable to have different types for elements and sums:
//       e.g., uint16_t may be sufficient for elements, but not for their total sum.
//       With floating-point elements, it's not uncommon to compute the sum at double precision.
template<Additive T>
class CumulativeHistogram {
 public:
   // Split capacity into M = ceil(Nmax/BucketSize) buckets, and build a tree for those. This
   // improves performance for small N for arithmetic types (by enabling vectorization).
  // TODO: make this a template parameter.
  static constexpr std::size_t BucketSize = 128;

  using value_type = T;
  // TODO: declare as `typename std::vector<T>::size_type`.
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = const T&;
  using const_reference = const T&;
  using pointer = const T*;
  using const_pointer = const T*;
  using iterator = const T*;
  using const_iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  // Constructs an empty histogram.
  // Time complexity: O(1).
  constexpr CumulativeHistogram() noexcept = default;

  // Copy constructor.
  // Note that this->capacity() == other.size() after construction.
  //
  // Time complexity: O(N), where N = other.size().
  constexpr CumulativeHistogram(const CumulativeHistogram& other);

  // Move constructor.
  // Note that the capacity of the constructed histogram is the same as what other.capacity() used to be.
  // After this call `other` is in a valid state, but both its size and its capacity are 0.
  //
  // Time complexity: O(1).
  constexpr CumulativeHistogram(CumulativeHistogram&& other) noexcept;

  // Copy assignment operator.
  // If this->capacity() >= other.size() before this call, then capacity remains the same.
  // Otherwise, capacity becomes equal to other.size().
  //
  // Time complexity: O(N + M), where N == this->size(), M == other.size().
  constexpr CumulativeHistogram& operator=(const CumulativeHistogram& other);

  // Move assignment operator.
  // After this call `other` is in a valid state, but both its size and its capacity are 0.
  //
  // Time complexity: O(N), where N == this->size() (because we need to destruct the currently
  // stored elements).
  constexpr CumulativeHistogram& operator=(CumulativeHistogram&& other)
    noexcept(std::is_nothrow_move_assignable_v<std::vector<T>>);

  // Destructor.
  // Time complexity: O(N), where N == this->size().
  constexpr ~CumulativeHistogram() = default;

  // Constructs a cumulative histogram for N zero-initialized elements.
  // Time complexity: O(N).
  explicit CumulativeHistogram(size_type num_elements);

  // Constructs a cumulative histogram for N elements, initializing them with the specified value.
  // Time complexity: O(N).
  explicit CumulativeHistogram(size_type num_elements, const T& value);

  // Constructs a cumulative histogram for the specified elements.
  // Note that this this->capacity() equals elements.size() after this call (not elements.capacity()).
  //
  // Time complexity: O(N).
  explicit CumulativeHistogram(std::vector<T>&& elements);

  // Constructs a cumulative histogram for the specified elements.
  // This overload only participates in overload resolution if Iter satisfies
  // std::input_iterator concept, to avoid ambiguity with CumulativeHistogram(size_type, T).
  // Time complexity: O(N), where N is the distance between first and last.
  template<std::input_iterator Iter>
  CumulativeHistogram(Iter first, Iter last);

  // Returns an iterator to the first element.
  constexpr const_iterator begin() const noexcept;

  // Returns an iterator past the last element.
  constexpr const_iterator end() const noexcept;

  // Returns a reverse iterator to the last element.
  constexpr const_reverse_iterator rbegin() const noexcept;

  // Returns a reverse iterator preceding the first element.
  constexpr const_reverse_iterator rend() const noexcept;

  // Returns true if the number of elements is 0, false otherwise.
  // Time complexity: O(1).
  constexpr bool empty() const noexcept;

  // Returns the number of elements in the histogram.
  // Time complexity: O(1).
  constexpr size_type size() const noexcept;

  // Returns the number of elements that can be held in currently allocated storage.
  // Time complexity: O(1).
  constexpr size_type capacity() const noexcept;

  // Reserves memory for a histogram capable of storing the specified number of elements.
  // The values of existing elements remain unchanged.
  // If reallocation takes place (i.e. if `this->capacity() < num_elements` before the call),
  // all references and iterators are invalidated.
  // Throws std::length_error or whatever std::vector::reserve() throws on failure.
  // If an exception is thrown, this function has no effect (strong exception guarantee).
  // Time complexity: O(1) if num_elements <= this->capacity(),
  //                  otherwise O(N), where N is the current number of elements.
  void reserve(size_type num_elements);

  // Erases all elements.
  // The capacity remains unchanged.
  // Time complexity: O(N).
  void clear() noexcept;

  // Add a zero-initialized element to the end.
  // Time complexity: amortized O(logN).
  // TODO: make it amortized O(1).
  void push_back();

  // Add an element to the end.
  // Time complexity: amortized O(logN).
  // TODO: make it amortized O(1).
  void push_back(const T& value);

  // Removes the last element.
  // Throws std::logic_error if this->empty().
  // Time complexity: O(1).
  void pop_back();

  // Changes the number of elements stored.
  // \param num_elements - the new number of elements in the histogram.
  // Time complexity: O(|N' - N|), if this->capacity() >= num_elements,
  //                  O(N') otherwise.
  void resize(size_type num_elements);

  // Sets the values of all elements to 0.
  // Time complexity: O(N).
  void setZero();

  // Sets the values of all elements to the specified value.
  // Time complexity: O(N).
  void fill(const T& value);

  // Access all elements.
  // Time complexity: O(1).
  constexpr std::span<const T> elements() const noexcept;

  // Access the specified element.
  // Throws std::out_of_range if k >= size().
  // Time complexity: O(1).
  const_reference element(size_type k) const;

  // Increment the specified element.
  // Throws std::out_of_range if k >= size().
  // Time complexity: O(log(N)).
  // TODO: remove the default value.
  void increment(size_type k, const T& value = 1);

  // Returns the k-th prefix sum of the stored elements.
  // Throws std::out_of_range if k >= size().
  // Time complexity: O(log(N)).
  T prefixSum(size_type k) const;

  // Returns the total sum of all elements.
  // Throws std::logic_error if this->empty().
  // Time complexity: O(log(N)).
  //
  // Note that it could be possible to implement this class so that totalSum() had O(1) time complexity:
  // for example, we could just add a member varaible and update it whenever push_back(), pop_back(),
  // increment() or resize() are called - updating the total sum would not affect the time complexity
  // of these operations.
  // However, it would unnecessarily affect performance in the use cases where the total sum is not needed
  // (or only needed rarely). Hence, I chose not to do it; if anything, it is very easy to write a wrapper
  // class that does it.
  T totalSum() const;

  // Find the first element, for which prefixSum() is not less than the specified value.
  // The behavior is undefined if any element is negative or if computing the total sum
  // causes an overflow for T.
  // \return { begin()+k, prefixSum(k) }, where k is the first element for which prefixSum(k) >= value,
  //         or { end(), T{} } if there is no such k.
  // Time complexity: O(log(N)).
  std::pair<const_iterator, T> lowerBound(const T& value) const;

  // Find the first element, for which prefixSum() is greater than the specified value.
  // The behavior is undefined if any element is negative or if computing the total sum
  // causes an overflow for T.
  // \return { begin()+k, prefixSum(k) }, where k is the first element for which prefixSum(k) > value,
  //         or { end(), T{} } if there is no such k.
  // Time complexity: O(log(N)).
  std::pair<const_iterator, T> upperBound(const T& value) const;

 private:
  // Returns an immutable TreeView for the currently effective tree.
  constexpr Detail_NS::TreeView<const T> getTreeView() const noexcept;

  // Returns an mutable TreeView for the currently effective tree.
  constexpr Detail_NS::TreeView<T> getMutableTreeView() noexcept;

  // Returns an immutable TreeViewSimple for the currently effective tree.
  constexpr Detail_NS::TreeViewSimple<const T> getTreeViewSimple() const noexcept;

  // Returns an immutable TreeViewSimple for the currently effective tree.
  constexpr Detail_NS::TreeViewSimple<T> getMutableTreeViewSimple() noexcept;

  // Returns the index of the effective root node.
  // * nodes_[getRootIndex()] stores the sum of elements from the left subtree -
  //   but only if there is a left subtree (i.e. if there current tree has at least 1 node).
  // Time complexity: O(1).
  constexpr size_type getRootIndex() const noexcept;

  // Returns the maximum number of elements that can represented by the current tree.
  //   num_elements <= capacityCurrent() <= capacity_
  constexpr size_type capacityCurrent() const noexcept;

  // Shared implementation for lowerBound() and upperBound().
  // For computing the lower bound, Compare should effectively implement `lhs < rhs`.
  // For computing the upper bound, Compare should effectively implement `lhs <= rhs`.
  // \return { begin()+k, prefixSum(k) }, where k is the first element for which !cmp(prefixSum(k), value)
  //      or { end(), T{} } if there is no such k.
  // Time complexity: O(log(N)).
  template<class Compare>
  std::pair<const_iterator, T> lowerBoundImpl(const T& value, Compare cmp) const;

  // Values of elements in the histogram.
  std::vector<T> elements_;
  // Nodes of the tree.
  // The number of nodes is always equal to Detail_NS::countNodesInBucketizedTree(countBuckets(capacity_, BucketSize));
  // TODO: only *construct* nodes that are needed to represent the currently effective tree.
  std::unique_ptr<T[]> nodes_;
  // Current capacity Nmax.
  // Nmax <= elements_.capacity(). We store the capacity explicitly because we want to double it when
  // calling push_back() at full capacity (to avoid rebuilding the tree), but std::vector doesn't
  // guarantee that it doubles the capacity.
  size_type capacity_ = 0;
};

// ======================================== Implementation ========================================

// CumulativeHistogram stores auxiliary counters for sums of certain elements.
// These counters are updated whenever the respective elements are modified -
// this is why CumulativeHistogram::increment() has O(log(N)) time complexity.
// However, having these sums precomputed also allows efficient computation of the prefix sums.
//
// These additional counters form an implicit binary tree:
// * The main tree represents all elements [0; N).
// * The left branch is a tree representing elements [0; (N-1)/2].
// * The right branch is a tree representing elements [(N+1)/2; N).
// * The tree is balanced: for any node the numbers of nodes in the left and right branches
//   differ at most by 1.
// * The nodes of the tree are stored as a plain array: [root, left0, left1, ... leftM, right0, right1, ... rightK].
// * The root stores the sum of elements from its left branch.
// * The tree only stores sums of elements, not the individual elements.
//
// For example, for N=11, the total number of nodes is countNodesInTree(11) = 6:
//         _ n0 __
//        /       \
//       n1        n4
//     /    \      /
//   n2     n3     n5
// ......imaginary nodes......
//  /  \    / \    /  \    \
// x0  x2  x3  x5 x6  x8   x9
//  \       \      \        \
//   x1      x4    x7      x10
//
// Nodes:
// +-------------------+----------+---------+-------+----------+---------+
// |    n0             |    n1    |   n2    |   n3  |    n4    |    n5   |
// +-------------------+----------+---------+-------+----------+---------+
// | x0+x1+x2+x3+x4+x5 | x0+x1+x2 | x0+x1   | x3+x4 | x6+x7+x8 | x6 + x7 |
// +-------------------+----------+---------+-------+----------+---------+
// Any prefix sum can be computed by going from the root of the tree to the leaf:
// +---------------------------------------------------------------------------------------------+
// |  s0  |  s1  |  s2  |  s3   |  s4   |  s5  |  s6   |  s7   |  s8   |  s9      |    s10       |
// +---------------------------------------------------------------------------------------------+
// |  x0  |  n2  |  n1  | n1+x3 | n1+n3 |  n0  | n0+x6 | n0+n5 | n0+n4 | n0+n4+x9 | n0+n4+x9+x10 |
// +---------------------------------------------------------------------------------------------+
//
// In order to support efficient insertion, the class grows in the same way that std::vector does:
// if size() == capacity(), then the next call to push_back() will allocate memory for a histogram
// capable of storing 2N elements.
//
// Note that the way the nodes are stored in a plain array allows to view any subtree as a subspan of
// that array. This comes in handy, allowing us to pretend that the deepest leftmost subtree that can
// represent all current elements *is* the tree. This way we ensure that the time complexity of the
// operations is actually O(logN), and not O(logNmax), where N is the number of elements and
// Nmax is capacity.

namespace Detail_NS {

  // Returns floor(log2(x)).
  // The result is unspecified if value == 0.
  constexpr std::size_t floorLog2(std::size_t value) noexcept {
    return std::bit_width(value) - 1;
  }

  // Returns ceil(log2(x)).
  // The result is unspecified if value == 0.
  constexpr std::size_t ceilLog2(std::size_t value) noexcept {
    const bool is_power_of_2 = std::has_single_bit(value);
    const std::size_t floor_log2 = floorLog2(value);
    return is_power_of_2 ? floor_log2 : (floor_log2 + 1);
  }

  // Returns the number of buckets needed to represent the specified number of elements.
  // \param num_elements - the number of elements.
  // \param bucket_size - the number of elements per bucket.
  // \returns ceil(num_elements / bucket_size).
  constexpr std::size_t countBuckets(std::size_t num_elements, std::size_t bucket_size) noexcept {
    return (num_elements / bucket_size) + (num_elements % bucket_size != 0);
  }

  // Returns the number of nodes in a tree representing the specified number of buckets.
  // \param num_buckets - the number of buckets represented by the tree.
  constexpr std::size_t countNodesInBucketizedTree(std::size_t num_buckets) noexcept {
    // f(0) = 0
    // f(1) = 0
    // for any N > 1:
    //   f(2N) = 1 + 2*f(N)
    //   f(2N+1) = 1 + f(N+1) + f(N)
    // It's easy to prove by induction that f(N) = N - 1 for any N > 1.
    return num_buckets == 0 ? 0 : (num_buckets - 1);
  }

  // Returns the number of elements represented by the leftmost subtree with root at the specified level.
  // \param num_elements - the total number of elements represented by the tree.
  // \param level - depth of the tree for which to count the number of elements. If level == 0, the
  //                function returns `num_elements`.
  // Time complexity: O(1).
  constexpr std::size_t countElementsInLeftmostSubtree(std::size_t num_elements, std::size_t level) noexcept {
    // First, note that h(x) = ceil(ceil(x/2)/2) = ceil(x/4). Proof:
    //   h(4a) = a
    //   h(4a+1) = ceil(ceil((4a+1)/2)/2) = ceil((2a+1)/2) = a+1
    //   h(4a+2) = ceil(ceil((4a+2)/2)/2) = ceil((2a+1)/2) = a+1
    //   h(4a+3) = ceil(ceil((4a+3)/2)/2) = ceil((2a+2)/2) = a+1
    //
    // The number of elements in the leftmost subtree is computed as:
    //   f(0) = N = ceil(N/1)
    //   f(1) = ceil(f(0)/2) = ceil(N/2)
    //   f(2) = ceil(f(1)/2) = ceil(ceil(N/2)/2) = ceil(N/4)
    //   f(3) = ceil(f(2)/2) = ceil(ceil(N/4)/2) = ceil(N/8)
    //   f(k) = ceil(N/2^k)
    const std::size_t mask = (static_cast<std::size_t>(1) << level) - 1;
    const std::size_t floored_result = num_elements >> level;
    const std::size_t remainder = num_elements & mask;
    return floored_result + (remainder != 0);
  }

  // Finds the deepest node containing all of the given elements in the tree with the specified capacity.
  // \param num_elements - specifies the range [0; num_elements) that must be contained by the node.
  // \param capacity - the total number of elements that can be represented by the tree.
  // \return the depth of the deepest node containing the elements [0; num_elements)
  //         in the optimal tree representing the elements [0; capacity),
  //         or static_cast<std::size_t>(-1) if capacity < num_elements.
  //         The depth of the root is 0.
  // Time complexity: O(1).
  constexpr std::size_t findDeepestNodeForElements(std::size_t num_elements,
                                                   std::size_t capacity) noexcept {
    // Let's assume that we have an optimal tree representing Nmax elements.
    // Its left subtree represents ceil(Nmax/2) elements.
    // The left subtree of the above-mentioned subtree represents ceil(ceil(Nmax/2)/2) = ceil(Nmax/4) elements.
    // and so on.
    //   x0 = Nmax, x1 = ceil(Nmax/2), ..., xK = ceil(Nmax/2^K)
    // We want to find the deepest subtree that represents at least N elements.
    // Which brings us to the inequality:
    //   ceil(Nmax/2^k) >= N
    // <=>
    //   Nmax/2^k + 1 > N
    // <=>
    //   Nmax/(N-1) > 2^k
    // <=>
    //   k < log2(Nmax/(N-1))
    // <=>
    //   The greatest such k is ceil(log2(Nmax/(N-1)))-1
    //
    // Edge cases:
    // * Nmax=0 is an edge case because log2(0) is undefined. In our case it means that
    //   the capacity of CumulativeHistogram is 0, so we return 0.
    // * Nmax=1 is an edge case because in this case N can be either 0 or 1 - both of
    //   which are edge cases (see below). If CumulativeHistogram can store 1 element
    //   at most, then the tree has 0 nodes, so we return 0.
    // * N=1 is an edge case because it causes division by 0. In our case it means that
    //   we want to find the deepest leftmost subtree that represents at least 1 element.
    //   Note that due to ceiling the formula above implies that k should be infinite.
    //   That's not what we want though - if N=1, then we want to simply find the deepest
    //   leftmost subtree (i.e. the first leftmost subtree that represents exactly 1 element).
    // * N=0 is an edge case because it will result into log2(-Nmax), which is undefined.
    //   In our case it means that we should return the deepest leftmost subtree, which is the
    //   same as calling findDeepestNodeForElements(1, Nmax).
    if (capacity < 2) return 0;
    if (num_elements < 2) {
      // Find the smallest k such that ceil(Nmax/2^k) == 1
      // which is simply ceil(log2(Nmax))
      return ceilLog2(capacity);
    }
    // Note that ceil(log2(x)) = ceil(log2(ceil(x)).
    const std::size_t floored_ratio = capacity / (num_elements - 1);
    const std::size_t remainder = capacity % (num_elements - 1);
    const std::size_t ratio = floored_ratio + (remainder != 0);  // ceil(Nmax/(N-1))
    return ceilLog2(ratio) - 1;
  }

  // Find the depth of the leftmost subtree that represents exactly the specified number of elements.
  // \param capacity_to_find - capacity of the subtree to find.
  // \param capacity - capacity of the full tree.
  // \return depth of the leftmost subtree that represents exactly `capacity_to_find` elements,
  //         or static_cast<std::size_t>(-1) if there is no such subtree.
  // Time complexity: O(1).
  constexpr std::size_t findLeftmostSubtreeWithExactCapacity(std::size_t capacity_to_find,
                                                             std::size_t capacity) noexcept {
    // TODO: comments are out of date.
    // Edge cases:
    // 1) A smaller tree cannot contain a larger tree.
    // 2) Our trees always have capacity >= 2. Trees of capacity == 1 are problematic - we forbid
    //    nodes with fewer than 2 elements, so a tree representing 1 element cannot be anyone's subtree.
    if (capacity < capacity_to_find || capacity_to_find < 1) {
      return static_cast<std::size_t>(-1);
    }
    // Find the deepest leftmost node of the new tree that contains at least `capacity_to_find` elements.
    const std::size_t level = findDeepestNodeForElements(capacity_to_find, capacity);
    // Get the number of elements that this node contains.
    const std::size_t num_elements_at_level = countElementsInLeftmostSubtree(capacity, level);
    // The old tree is a subtree of the new one if and only if the number of elements matches exactly.
    if (num_elements_at_level == capacity_to_find) {
      return level;
    }
    return static_cast<std::size_t>(-1);
  }

  // Computes the new capacity for a full tree that currently stores the specified number of elements.
  template<std::size_t BucketSize>
  constexpr std::size_t computeNewCapacityForFullTree(std::size_t capacity) noexcept {
    // Special case: if the tree is currently empty, then the new capacity is simply 2.
    if (capacity == 0) {
      return 2;
    }
    // Otherwise, there's at least 1 bucket already. Check if 2*Nmax elements still fit into a single bucket;
    // if so, just double the current capacity - the number of buckets will remain 1.
    if (capacity * 2 <= BucketSize) {
      return capacity * 2;
    }
    // Otherwise, double the number of buckets.
    // Compute the number of buckets in the current tree.
    const std::size_t num_buckets = countBuckets(capacity, BucketSize);
    // What is the smallest M such that
    //     ceil(M/BucketSize) = 2*ceil(Nmax/BucketSize)
    //   AND
    //     M >= Nmax*2
    // ?
    // The first inequality gives the lower and the upper bounds:
    //   2*ceil(Nmax/BucketSize) * BucketSize - (BucketSize - 1) <= M <= 2*ceil(Nmax/BucketSize) * BucketSize
    // Therefore, the answer is
    //   max(2*ceil(Nmax/BucketSize) * BucketSize - (BucketSize - 1), Nmax*2)
    return std::max((2 * num_buckets - 1) * BucketSize + 1, capacity * 2);
  }

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
    // Index of the first element represented by the tree.
    std::size_t bucket_first_;
    // The number of real elements represented by the tree.n
    std::size_t num_buckets_;
    // The maximum number of elements this tree can represent.
    std::size_t bucket_capacity_;
  };

  // High-level API for interacing with an implicit tree data structure.
  template<class T>
  class TreeView : public TreeViewBase {
  public:
    // Expects that 0 < num_elements <= capacity.
    constexpr TreeView(std::span<T> nodes,
                       std::size_t num_elements,
                       std::size_t capacity) noexcept :
      TreeViewBase(num_elements, capacity),
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

  // Non-template base for TreeViewSimple.
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
    // The number of buckets represented by the tree.n
    std::size_t num_buckets_;
  };

  template<class T>
  class TreeViewSimple : public FullTreeViewBase {
   public:
    constexpr TreeViewSimple(T* root, std::size_t num_buckets) noexcept:
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

    constexpr explicit CompressedPath(std::size_t bucket_capacity):
      bucket_capacity_(bucket_capacity)
    {}

    constexpr std::span<const Entry> path() const noexcept { return path_; }

    // Appends a bucket to the end.
    // Time complexity: O(1).
    void pushBack() {
      // TODO: double bucket_capacity if the tree is full.
      // If this is the first bucket, just add 1 entry.

      // Special case - adding the first bucket.
      if (path_.empty()) {
        // If the path is empty, it means that the current tree has 0 nodes.
        // The new entry will represent the immediate left subtree of the root.
        path_.push_back(Entry{ .node = PathEntry(1, 1), .level = 1, .is_left_subtree = true });
        return;
      }

      const Entry& last_entry = path_.back();
      if (last_entry.is_left_subtree) {
        const std::size_t level_was = last_entry.level;
        if (level_was == 1) {
          // Delete the last entry.
          path_.pop_back();
          // Check if there is a previous entry at all.
          if (path_.empty()) {
            // Add an entry for the leftmost subtree of the immediate right subtree of the root.
            // We know that the immediate right subtree of the root has 0 nodes, because the deleted
            // entry was the immediate left subtree, but it was also a leaf, which means that the height
            // of the right subtree cannot be greater than 1.
            // PathEntry(1, 1) is the same as getRootEntry().rightChild().
            path_.push_back(Entry{ .node = PathEntry(1, 1), .level = 1, .is_left_subtree = false });
          }
          else {
            // Otherwise, the previous entry must be some right subtree of depth M.
            assert(path_.back().is_left_subtree == false);
            // Replace the previous entry with an entry for its immediate right subtree.
            Entry& new_last_entry = path_.back();
            new_last_entry.node.switchToRightChild();
            // Increment the level from M to M+1.
            ++new_last_entry.level;
          }
        }
        else {
          // TODO: there's no need to need to construct the root entry if the path has more than 1 element.
          // Actually, this should never happen. If the last entry is the leftmost subtree at level K of the root,
          // then K must be 1.
          //const PathEntry root_entry = getRootEntry();
          // Remove the last entry for the K-th leftmost subtree of some node.
          path_.pop_back();
          assert(!path_.empty());
          // Add a new entry for the (K-1)-th leftmost subtree of the same node.
          path_.push_back(Entry{ .node = path_.back().node.leftmostChild(level_was - 1),
                                 .level = level_was - 1,
                                 .is_left_subtree = true });
          // Add another entry for the immediate right subtree of the node we've just added.
          path_.push_back(Entry{ .node = path_.back().node.rightChild(), .level = 1, .is_left_subtree = false });
        }
      }
      else {
        // OK, the last entry was the rightmost subtree of some node.
      }
    }

    // Removes the last bucket.
    // Time complexity: O(1).
    void popBack();

   private:
    // Returns the current number of buckets in the tree.
    std::size_t numBuckets() const noexcept {
      if (path_.empty()) {
        return 0;
      }
      const Entry& last_entry = path_.back();
      return last_entry.node.bucketFirst() + last_entry.node.numBuckets();
    }

    // Root itself is not stored in the path, but we need to use it sometimes.
    PathEntry getRootEntry() const noexcept {
      const std::size_t num_buckets_current = numBuckets();
      const std::size_t root_level = findDeepestNodeForElements(num_buckets_current, bucket_capacity_);
      const std::size_t bucket_capacity_current = countElementsInLeftmostSubtree(bucket_capacity_, root_level);
      return PathEntry(0, bucket_capacity_current);
    }

    std::vector<Entry> path_;
    std::size_t bucket_capacity_;
  };

  // This function is intended to be called for trees at their full capacity.
  // Computes the total sum of elements of a tree which is at its full capacity.
  // \param elements - elements represented by the tree.
  // \param nodes - node of the tree.
  // The behavior is undefined if
  //   elements.empty() || nodes.size() != countNodesInBucketizedTree(countBuckets(elements.size(), BucketSize))
  // \returns the total sum of elements from `elements`.
  // Time complexity: O(logN), where N = elements.size().
  template<class T, std::size_t BucketSize>
  constexpr T sumElementsOfFullTree(std::span<const T> elements, std::span<const T> nodes) {
    assert(!elements.empty());
    const std::size_t num_buckets = countBuckets(elements.size(), BucketSize);
    assert(nodes.size() == countNodesInBucketizedTree(num_buckets));
    TreeViewSimple<const T> tree(nodes.data(), num_buckets);
    T result{};
    while (!tree.empty()) {
      result += tree.root();
      tree.switchToRightChild();
    }
    // The last bucket contains at most BucketSize elements.
    // TODO: AFAIU, tree.bucketFirst() == num_buckets - 1, so there's no need to track it.
    const std::size_t first = tree.bucketFirst() * BucketSize;
    return std::accumulate(elements.begin() + first, elements.end(), std::move(result));
  }

  // Initializes the nodes of the specified tree according to the values of the given elements.
  // \param elements - values of elements for which we want to track prefix sums.
  // \param tree - tree for some or all elements from `elements`.
  // \param bucket_size - the number of elements per bucket.
  // \returns the total sum of elements represented by `tree`.
  template<class T>
  T buildBucketizedTreeImpl(std::span<const T> elements, const TreeView<T>& tree, std::size_t bucket_size) {
    TreeView<T> t = tree;
    T total_sum{};
    while (!t.empty()) {
      T total_sum_left = buildBucketizedTreeImpl(elements, t.leftChild(), bucket_size);
      total_sum += std::as_const(total_sum_left);
      t.root() = std::move(total_sum_left);
      t.switchToRightChild();
    }
    assert(t.numBuckets() == 1);
    const std::size_t bucket_index = t.bucketFirst();
    const std::size_t element_first = bucket_index * bucket_size;
    const std::size_t num_elements = std::min(elements.size() - element_first, bucket_size);
    const std::span<const T> elements_in_bucket = elements.subspan(element_first, num_elements);
    return std::accumulate(elements_in_bucket.begin(), elements_in_bucket.end(), std::move(total_sum));
  }

  // Builds the tree for the given elements.
  // Expects that:
  // 1) 0 <= elements.size() <= capacity
  // 2) nodes.size() == countNodesInBucketizedTree(countBuckets(capacity, bucket_size))
  // Time complexity: O(N), where N = elements.size().
  // TODO: change the API so that `nodes` is only required to have enough nodes to represent all elements.
  //       Or just pass const std::vector& and let buildTree() decide the optimal structure.
  template<class T>
  void buildBucketizedTree(std::span<const T> elements, std::span<T> nodes, std::size_t capacity, std::size_t bucket_size) {
    if (elements.empty()) {
      return;
    }
    const std::size_t bucket_capacity = countBuckets(capacity, bucket_size);
    const std::size_t max_num_nodes = countNodesInBucketizedTree(bucket_capacity);
    assert(nodes.size() == max_num_nodes);
    const std::size_t num_buckets = countBuckets(elements.size(), bucket_size);

    const std::size_t level = findDeepestNodeForElements(num_buckets, bucket_capacity);
    const std::size_t bucket_capacity_at_level = countElementsInLeftmostSubtree(bucket_capacity, level);
    const std::size_t num_nodes_at_level = countNodesInBucketizedTree(bucket_capacity_at_level);
    if (num_nodes_at_level == 0) {
      return;
    }
    const std::span<T> nodes_at_level = nodes.subspan(level, num_nodes_at_level);
    TreeView<T> tree(nodes_at_level, num_buckets, bucket_capacity_at_level);
    buildBucketizedTreeImpl(elements, tree, bucket_size);
  }

}  // namespace Detail_NS

template<Additive T>
constexpr Detail_NS::TreeView<const T> CumulativeHistogram<T>::getTreeView() const noexcept {
  // The maximum number of buckets the current tree can represent.
  const std::size_t bucket_capacity = Detail_NS::countBuckets(capacity(), BucketSize);
  // Number of buckets needed to represent the current elements.
  const std::size_t num_buckets = Detail_NS::countBuckets(elements_.size(), BucketSize);
  // Level of the currently effective tree.
  const std::size_t root_level = Detail_NS::findDeepestNodeForElements(num_buckets, bucket_capacity);
  // The number of buckets at the current level.
  const std::size_t bucket_capacity_at_level = Detail_NS::countElementsInLeftmostSubtree(bucket_capacity, root_level);
  // The number of nodes at the current level.
  const std::size_t num_nodes_at_level = Detail_NS::countNodesInBucketizedTree(bucket_capacity_at_level);
  return Detail_NS::TreeView<const T> {
    std::span<const T> { nodes_.get() + root_level, num_nodes_at_level },
      num_buckets, bucket_capacity_at_level
  };
}

template<Additive T>
constexpr Detail_NS::TreeView<T> CumulativeHistogram<T>::getMutableTreeView() noexcept {
  // The maximum number of buckets the current tree can represent.
  const std::size_t bucket_capacity = Detail_NS::countBuckets(capacity(), BucketSize);
  // Number of buckets needed to represent the current elements.
  const std::size_t num_buckets = Detail_NS::countBuckets(elements_.size(), BucketSize);
  // Level of the currently effective tree.
  const std::size_t root_level = Detail_NS::findDeepestNodeForElements(num_buckets, bucket_capacity);
  // The number of buckets at the current level.
  const std::size_t bucket_capacity_at_level = Detail_NS::countElementsInLeftmostSubtree(bucket_capacity, root_level);
  // The number of nodes at the current level.
  const std::size_t num_nodes_at_level = Detail_NS::countNodesInBucketizedTree(bucket_capacity_at_level);
  return Detail_NS::TreeView<T> {
    std::span<T> { nodes_.get() + root_level, num_nodes_at_level },
      num_buckets, bucket_capacity_at_level
  };
}

template<Additive T>
constexpr Detail_NS::TreeViewSimple<const T> CumulativeHistogram<T>::getTreeViewSimple() const noexcept {
  // The maximum number of buckets the current tree can represent.
  const std::size_t bucket_capacity = Detail_NS::countBuckets(capacity(), BucketSize);
  // Number of buckets needed to represent the current elements.
  const std::size_t num_buckets = Detail_NS::countBuckets(elements_.size(), BucketSize);
  // Level of the currently effective tree.
  const std::size_t root_level = Detail_NS::findDeepestNodeForElements(num_buckets, bucket_capacity);
  // The number of buckets at the current level.
  const std::size_t bucket_capacity_at_level = Detail_NS::countElementsInLeftmostSubtree(bucket_capacity, root_level);
  return Detail_NS::TreeViewSimple<const T> {nodes_.get() + root_level, bucket_capacity_at_level};
}

template<Additive T>
constexpr Detail_NS::TreeViewSimple<T> CumulativeHistogram<T>::getMutableTreeViewSimple() noexcept {
  // The maximum number of buckets the current tree can represent.
  const std::size_t bucket_capacity = Detail_NS::countBuckets(capacity(), BucketSize);
  // Number of buckets needed to represent the current elements.
  const std::size_t num_buckets = Detail_NS::countBuckets(elements_.size(), BucketSize);
  // Level of the currently effective tree.
  const std::size_t root_level = Detail_NS::findDeepestNodeForElements(num_buckets, bucket_capacity);
  // The number of buckets at the current level.
  const std::size_t bucket_capacity_at_level = Detail_NS::countElementsInLeftmostSubtree(bucket_capacity, root_level);
  return Detail_NS::TreeViewSimple<T> {nodes_.get() + root_level, bucket_capacity_at_level};
}

template<Additive T>
constexpr
typename CumulativeHistogram<T>::size_type
CumulativeHistogram<T>::getRootIndex() const noexcept {
  // The maximum number of buckets the current tree can represent.
  const std::size_t bucket_capacity = Detail_NS::countBuckets(capacity(), BucketSize);
  // Number of buckets needed to represent the current elements.
  const std::size_t num_buckets = Detail_NS::countBuckets(elements_.size(), BucketSize);
  return Detail_NS::findDeepestNodeForElements(num_buckets, bucket_capacity);
}

template<Additive T>
constexpr
typename CumulativeHistogram<T>::size_type
CumulativeHistogram<T>::capacityCurrent() const noexcept {
  // The maximum number of buckets the current tree can represent.
  const std::size_t bucket_capacity = Detail_NS::countBuckets(capacity(), BucketSize);
  // Number of buckets needed to represent the current elements.
  const std::size_t num_buckets = Detail_NS::countBuckets(elements_.size(), BucketSize);
  const std::size_t level = Detail_NS::findDeepestNodeForElements(num_buckets, bucket_capacity);
  if (level == 0) {
    return capacity();
  }
  const std::size_t bucket_capacity_at_level = Detail_NS::countElementsInLeftmostSubtree(bucket_capacity, level);
  return bucket_capacity_at_level * BucketSize;
}

template<Additive T>
constexpr CumulativeHistogram<T>::CumulativeHistogram(const CumulativeHistogram& other):
  CumulativeHistogram(other.begin(), other.end())
{}

template<Additive T>
constexpr CumulativeHistogram<T>::CumulativeHistogram(CumulativeHistogram&& other) noexcept :
  elements_(std::move(other.elements_)),
  nodes_(std::move(other.nodes_)),
  capacity_(std::exchange(other.capacity_, static_cast<size_type>(0)))
{}

template<Additive T>
constexpr CumulativeHistogram<T>& CumulativeHistogram<T>::operator=(const CumulativeHistogram& other)
{
  if (capacity_ < other.size()) {
    // Delegate to copy constructor and move assignment operator.
    return *this = CumulativeHistogram(other);
  }
  // Our capacity is sufficient to store all elements from `other`, so no memory allocation is needed.
  // TODO: check the special case when the other tree can be copied.
  elements_.clear();
  elements_.insert(elements_.end(), other.begin(), other.end());
  const size_type num_buckets = Detail_NS::countBuckets(capacity_, BucketSize);
  const size_type num_nodes = Detail_NS::countNodesInBucketizedTree(num_buckets);
  const std::span<T> nodes{ nodes_.get(), num_nodes };
  Detail_NS::buildBucketizedTree<T>(elements_, nodes, capacity_, BucketSize);
  return *this;
}

template<Additive T>
constexpr CumulativeHistogram<T>& CumulativeHistogram<T>::operator=(CumulativeHistogram&& other)
  noexcept(std::is_nothrow_move_assignable_v<std::vector<T>>)
{
  // This line might throw an exception.
  elements_ = std::move(other.elements_);
  // These lines cannot throw any exceptions.
  nodes_ = std::move(other.nodes_);
  capacity_ = std::exchange(other.capacity_, static_cast<size_type>(0));
  return *this;
}

template<Additive T>
CumulativeHistogram<T>::CumulativeHistogram(size_type num_elements):
  elements_(num_elements),
  capacity_(num_elements)
{
  const size_type num_buckets = Detail_NS::countBuckets(num_elements, BucketSize);
  const size_type num_nodes = Detail_NS::countNodesInBucketizedTree(num_buckets);
  if (num_nodes != 0) {
    // Allocate and zero-initialize the nodes.
    nodes_ = std::make_unique<T[]>(num_nodes);
  }
}

template<Additive T>
CumulativeHistogram<T>::CumulativeHistogram(size_type num_elements, const T& value):
  CumulativeHistogram(std::vector<T>(num_elements, value))
{
}

template<Additive T>
CumulativeHistogram<T>::CumulativeHistogram(std::vector<T>&& elements):
  elements_(std::move(elements)),
  capacity_(elements_.size())
{
  // TODO: only construct nodes that are needed to represent the current level.
  const size_type num_buckets = Detail_NS::countBuckets(capacity_, BucketSize);
  const size_type num_nodes = Detail_NS::countNodesInBucketizedTree(num_buckets);
  if (num_nodes != 0) {
    // Allocate and default-initialize the nodes - there's no need to zero-initialize them.
    nodes_ = std::make_unique_for_overwrite<T[]>(num_nodes);
    const std::span<T> nodes{ nodes_.get(), num_nodes };
    Detail_NS::buildBucketizedTree<T>(elements_, nodes, capacity_, BucketSize);
  }
}

template<Additive T>
template<std::input_iterator Iter>
CumulativeHistogram<T>::CumulativeHistogram(Iter first, Iter last):
  CumulativeHistogram(std::vector<T>(first, last))
{
}

template<Additive T>
constexpr
typename CumulativeHistogram<T>::const_iterator CumulativeHistogram<T>::begin() const noexcept {
  return elements_.data();
}

template<Additive T>
constexpr
typename CumulativeHistogram<T>::const_iterator CumulativeHistogram<T>::end() const noexcept {
  return elements_.data() + size();
}

template<Additive T>
constexpr
typename CumulativeHistogram<T>::const_reverse_iterator CumulativeHistogram<T>::rbegin() const noexcept {
  return std::make_reverse_iterator(end());
}

template<Additive T>
constexpr
typename CumulativeHistogram<T>::const_reverse_iterator CumulativeHistogram<T>::rend() const noexcept {
  return std::make_reverse_iterator(begin());
}

template<Additive T>
constexpr bool CumulativeHistogram<T>::empty() const noexcept {
  return elements_.empty();
}

template<Additive T>
constexpr
typename CumulativeHistogram<T>::size_type
CumulativeHistogram<T>::size() const noexcept {
  return elements_.size();
}

template<Additive T>
constexpr
typename CumulativeHistogram<T>::size_type
CumulativeHistogram<T>::capacity() const noexcept {
  return capacity_;
}

template<Additive T>
void CumulativeHistogram<T>::reserve(size_type num_elements) {
  if (num_elements <= capacity()) {
    return;
  }

  // Get the capacity of the currently effective tree.
  const size_type capacity_current = capacityCurrent();
  // The maximum number of buckets the currently effective tree can represent.
  const std::size_t bucket_capacity = Detail_NS::countBuckets(capacity_current, BucketSize);
  // The maximum number of buckets the new tree can represent.
  const std::size_t bucket_capacity_new = Detail_NS::countBuckets(num_elements, BucketSize);
  // Compute the number of nodes in the new tree.
  const std::size_t num_nodes_new = Detail_NS::countNodesInBucketizedTree(bucket_capacity_new);

  // Special case - if the new tree has 0 nodes, then that means that the original tree also has 0 nodes.
  if (num_nodes_new == 0) {
    elements_.reserve(num_elements);
    capacity_ = num_elements;
    return;
  }

  // Allocate memory for the new tree.
  // TODO: only construct the nodes that are needed to represent the current level.
  std::unique_ptr<T[]> new_nodes = std::make_unique_for_overwrite<T[]>(num_nodes_new);
  const std::span<T> new_nodes_span{ new_nodes.get(), num_nodes_new };

  // Check the special case when the tree for num_elements has our current tree as a subtree.
  // In that case there's no need to rebuild the tree - we can just copy our current one.
  const std::size_t level_for_the_original =
    Detail_NS::findLeftmostSubtreeWithExactCapacity(bucket_capacity, bucket_capacity_new);
  // Construct the new tree.
  if (level_for_the_original == static_cast<std::size_t>(-1)) {
    // The old tree is not a subtree of the new tree, so we have to build the new one from scratch.
    Detail_NS::buildBucketizedTree<T>(elements_, new_nodes_span, num_elements, BucketSize);
    // Reserve new data for elements.
    elements_.reserve(num_elements);
  } else {
    // Just copy the current tree as a subtree of the new one.
    const size_type root_idx_old = getRootIndex();
    const size_type num_nodes_old = Detail_NS::countNodesInBucketizedTree(bucket_capacity);
    const std::span<T> effective_nodes_old { nodes_.get() + root_idx_old, num_nodes_old };
    const std::span<T> effective_nodes_new = new_nodes_span.subspan(level_for_the_original, num_nodes_old);
    // Basic exception guarantee: we only move the nodes if T's move assignment is noexcept;
    // otherwise, we copy them, so that even if an exception is thrown during copying, this class
    // will remain in a valid state.
    // TODO: replace the condition with std::std::is_nothrow_constructible_v after implementing
    // lifetimes for nodes. Note that in this case you won't be able to simply copy all num_nodes_old -
    // some of them may not have started their lifetime yet.
    if constexpr (std::is_nothrow_move_assignable_v<T>) {
      // Memory is reserved before we move the nodes to ensure strong exception guarantee:
      // std::vector::reserve() may throw an exception, but std::copy_n() cannot.
      elements_.reserve(num_elements);
      std::copy_n(std::make_move_iterator(effective_nodes_old.begin()), effective_nodes_old.size(),
                  effective_nodes_new.begin());
    } else {
      std::copy_n(effective_nodes_old.cbegin(), effective_nodes_old.size(),
                  effective_nodes_new.begin());
      // To ensure strong exception guarantee, std::vector::reserve() must be called after (potentitally
      // throwing) std::copy_n(): if reserve() succeeds, then elements_ will be modified.
      elements_.reserve(num_elements);
    }
  }
  // Replace old data with new data.
  nodes_ = std::move(new_nodes);
  capacity_ = num_elements;
}

template<Additive T>
void CumulativeHistogram<T>::clear() noexcept {
  elements_.clear();
  // TODO: destroy the nodes.
}

template<Additive T>
void CumulativeHistogram<T>::setZero() {
  const T zero {};
  std::fill(elements_.begin(), elements_.end(), zero);
  // Zero-out the active nodes.
  Detail_NS::TreeView<T> tree = getMutableTreeView();
  // Shortcut: if the current tree is at full capacity, just zero-out all
  // of its nodes and return.
  if (tree.numBuckets() == tree.bucketCapacity()) {
    const std::span<T> nodes = tree.nodes();
    std::fill_n(nodes.data(), nodes.size(), zero);
    return;
  }
  while (!tree.empty()) {
    const std::span<T> nodes = tree.nodes();
    // The left subtree is always full, therefore all its nodes are active.
    const std::size_t num_nodes_left = nodes.size() / 2;  // ceil((nodes.size() - 1) / 2)
    // Zero-initialize the root and the nodes of the left subtree.
    std::fill_n(nodes.data(), 1 + num_nodes_left, zero);
    // Switch to the effective right subtree.
    tree.switchToRightChild();
  }
}

template<Additive T>
void CumulativeHistogram<T>::push_back() {
  push_back(T{});
}

template<Additive T>
void CumulativeHistogram<T>::push_back(const T& value) {
  // Double the capacity if needed.
  if (size() == capacity()) {
    reserve(Detail_NS::computeNewCapacityForFullTree<BucketSize>(capacity()));
  }
  // Special case for the empty histogram.
  if (empty()) {
    elements_.push_back(value);
    return;
  }
  // There are 2 possibilities - either we can simply add the new element without constructing
  // any new nodes, or we need to extend some subtree by constructing a new root.
  Detail_NS::TreeView<const T> tree = getTreeView();
  // TODO: get rid of this loop. Traversing the tree has O(logN) time complexity, but it can be avoided
  // if we store the path to the deepest rightmost subtree. In that case updating the path can be done in O(1).
  // Loop until we find a full tree, i.e. the tree whose last theoretical element is N-1, where N is
  // the current number of elements.
  while ((tree.bucketFirst() + tree.bucketCapacity()) * BucketSize != size()) {
    // If the tree has no nodes (which means it can represent at most BucketSize elements) and is not at full capacity,
    // we can simply add the new element without constructing any new nodes.
    if (tree.empty()) {
      elements_.push_back(value);
      return;
    }
    // Otherwise, the left subtree must be full, so we switch to the effective right subtree.
    tree.switchToRightChild();
  }
  // If we are here, then we have found some non-empty subtree (maybe the main tree) that is at full capacity.
  // This subtree cannot be the *immediate* right subtree of some existing tree, because that would mean that
  // this tree is also at full capacity, which contradicts the loop condition - we were searching for the topmost
  // full subtree.
  // Hence, this subtree is NOT the *immediate* right subtree, i.e. it's the left subtree (at some depth K) of the
  // *immediate* right subtree. We will construct a new root, whose left subtree will be the tree we've found, and
  // the right subtree will store the element that we are adding.
  // TreeView doesn't store the index of the root node, but it's easy to compute via pointer arithmetic.
  const std::size_t tmp_root_idx = tree.nodes().data() - nodes_.get();
  const std::size_t root_idx_new = tmp_root_idx - 1;
  // Compute the sum of all elements in the effective right subtree.
  // This has O(logN) time complexity in the worst case, but, fortunately, the amortized time complexity is O(1).
  const std::size_t element_first = tree.bucketFirst() * BucketSize;
  const std::span<const T> tmp_elements = std::span<const T>(elements_).subspan(element_first);
  // Construct the new node.
  nodes_[root_idx_new] = Detail_NS::sumElementsOfFullTree<T, BucketSize>(tmp_elements, tree.nodes());
  // The new element is added to the right subtree of the newly constructed tree. This subtree doesn't
  // have any nodes yet, because it only represents 1 element.
  elements_.push_back(value);
}

template<Additive T>
void CumulativeHistogram<T>::pop_back() {
  if (empty()) {
    throw std::logic_error("CumulativeHistogram::pop_back(): there are no elements left to remove.");
  }
  // TODO: find the deepest rightmost subtree. If we are removing the only element from that subtree,
  // then we should destroy its *effective* root node.
  // Currently, though, we don't construct/destroy nodes, so whatever.
  elements_.pop_back();
}

template<Additive T>
void CumulativeHistogram<T>::resize(size_type num_elements) {
  // Do nothing if N == N'
  if (size() == num_elements) {
    return;
  }
  // Remove the last N-N' elements if N > N'.
  if (size() > num_elements) {
    // Lol, yes it works.
    // TODO: destoy the nodes that are no longer needed. Currenlty this is not needed because
    // we don't construct/destroy nodes manually.
    elements_.resize(num_elements);
    return;
  }
  // Append N'-N elements if N < N'.
  // TODO: reserve space for 2x more elements (or 4x or however much is needed),
  // so that we don't have to rebuild the tree.
  // Pros: resize will be slighly faster, because the tree can be copied instead of rebuilding.
  // Cons: memory overhead if extra space will not be used.
  // reserve(num_elements);
  // TODO: I think a more efficient implementation is possible: basically buildTree(), which only
  // updates new nodes.
  const size_type elements_to_add = num_elements - size();
  for (size_type i = 0; i < elements_to_add; ++i) {
    push_back(T{});
  }
}

template<Additive T>
constexpr std::span<const T> CumulativeHistogram<T>::elements() const noexcept {
  return std::span<const T>{elements_.data(), size()};
}

template<Additive T>
typename CumulativeHistogram<T>::const_reference
CumulativeHistogram<T>::element(size_type k) const {
  return elements_.at(k);
}

template<Additive T>
void CumulativeHistogram<T>::increment(size_type k, const T& value) {
  if (k >= size()) {
    throw std::out_of_range("CumulativeHistogram::increment(): k is out of range.");
  }
  // We are using TreeViewSimple here even though TreeView can skip inactive nodes in O(1)
  // time. The reason is that traversing TreeViewSimple is much faster - switching to the
  // left/right subtree only requires 3 or 4 instructions, but for TreeView it's much more.
  // This is essentially the constant factor of our O(logN) time complexity, and by
  // using TreeViewSimple we improve the average time complexity, whereas TreeView
  // optimizes the best-case time complexity at the cost of greater average time complexity.
  const size_type k_plus_one = k + 1;
  // Tree representing the elements [0; N).
  Detail_NS::TreeViewSimple<T> tree = getMutableTreeViewSimple();
  while (!tree.empty()) {
    // The root of the tree stores the sum of all elements [first; middle).
    const std::size_t middle = tree.pivot() * BucketSize;
    if (k_plus_one > middle) {
      tree.switchToRightChild();
    }
    else {
      // The root stores the sum of all elements in the left subtree, i.e. [first; middle),
      // so we should increment it if the root node is active - which is only if the right subtree
      // is not empty. The elements represented by the right subtree are [middle; size()), i.e.
      // it's not empty if and only if middle < size().
      if (middle < size()) {
        tree.root() += value;
      }
      // Break if k == middle-1: this implies that no other node contains elements_[k] as a term.
      if (k_plus_one == middle) {
        break;
      }
      tree.switchToLeftChild();
    }
  }
  // Update the element itself.
  elements_[k] += value;
}

template<Additive T>
T CumulativeHistogram<T>::prefixSum(size_type k) const {
  if (k >= size()) {
    throw std::out_of_range("CumulativeHistogram::prefixSum(): k is out of range.");
  }
  const size_type k_plus_one = k + 1;
  // Special case for the total sum
  if (k_plus_one == size()) {
    return totalSum();
  }
  T result {};
  // Tree representing the elements [0; N).
  Detail_NS::TreeViewSimple<const T> tree = getTreeViewSimple();
  while (!tree.empty()) {
    // The root of the tree stores the sum of all elements [first; middle).
    const std::size_t middle = tree.pivot() * BucketSize;
    if (k_plus_one < middle) {
      tree.switchToLeftChild();
    }
    else {
      result += tree.root();
      if (k_plus_one == middle) {
        return result;
      }
      tree.switchToRightChild();
    }
  }
  // Add elements from the bucket.
  const size_type first = tree.bucketFirst() * BucketSize;
  return std::accumulate(elements_.begin() + first, elements_.begin() + k_plus_one, std::move(result));
}

template<Additive T>
T CumulativeHistogram<T>::totalSum() const {
  if (empty()) {
    throw std::logic_error("CumulativeHistogram::totalSum(): the histogram is empty.");
  }
  T result {};
  Detail_NS::TreeView<const T> tree = getTreeView();
  while (!tree.empty()) {
    result += tree.root();
    tree.switchToRightChild();
  }
  // Add values of existing elements from the last bucket.
  const size_type first = tree.bucketFirst() * BucketSize;
  return std::accumulate(elements_.begin() + first, elements_.end(), std::move(result));
}

template<Additive T>
template<class Compare>
std::pair<typename CumulativeHistogram<T>::const_iterator, T>
CumulativeHistogram<T>::lowerBoundImpl(const T& value, Compare cmp) const {
  // Terminate if there are no elements.
  if (empty()) {
    return { end(), T{} };
  }
  T prefix_sum_before_lower {};
  T prefix_sum_upper {};
  // Tree representing the elements [k_lower; k_upper] = [0; N-1].
  Detail_NS::TreeView<const T> tree = getTreeView();
  while (!tree.empty()) {
    // The root of the tree stores the sum of all elements [k_lower; middle].
    // Sum of elements [0; middle]
    T prefix_sum_middle = prefix_sum_before_lower + tree.root();
    if (cmp(prefix_sum_middle, value)) {
      // OK, we don't need to check the left tree, because prefixSum(i) < value for i in [0; middle].
      // k_lower = middle + 1;
      prefix_sum_before_lower = std::move(prefix_sum_middle);
      tree.switchToRightChild();
    } else {
      // No need to check the right tree because prefixSum(i) >= value for i in [middle; N).
      // Note that it's still possible that middle is the element we're looking for.
      // k_upper = middle;
      prefix_sum_upper = std::move(prefix_sum_middle);
      tree.switchToLeftChild();
    }
  }
  // We know that cmp(prefixSum(i), value) == true for all i < k_lower
  const std::size_t k_lower = tree.bucketFirst() * BucketSize;
  // if k_upper_theoretical < size(), then cmp(prefixSum(i), value) == false for all i >= k_upper_theoretical
  const std::size_t k_upper_theoretical = k_lower + BucketSize;
  const std::size_t k_upper = std::min(k_upper_theoretical, size());
  T prefix_sum = std::move(prefix_sum_before_lower);
  for (std::size_t i = k_lower; i < k_upper; ++i) {
    prefix_sum += elements_[i];
    if (!cmp(prefix_sum, value)) {
      return { begin() + i, std::move(prefix_sum) };
    }
  }
  if (k_upper_theoretical < size()) {
    return { begin() + k_upper_theoretical, prefix_sum_upper };
  }
  return { end(), T{} };
}

template<Additive T>
std::pair<typename CumulativeHistogram<T>::const_iterator, T>
CumulativeHistogram<T>::lowerBound(const T& value) const {
  return lowerBoundImpl(value, std::less<T>{});
}

template<Additive T>
std::pair<typename CumulativeHistogram<T>::const_iterator, T>
CumulativeHistogram<T>::upperBound(const T& value) const {
  // Effectively implements `lhs <= rhs`, but only requires operator< to be defined for T.
  auto less_equal = [](const T& lhs, const T& rhs) { return !(rhs < lhs); };
  return lowerBoundImpl(value, less_equal);
}

}  // namespace CumulativeHistogram_NS
