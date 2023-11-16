#pragma once

#include <cumulative_histogram/BucketSize.h>
#include <cumulative_histogram/CompressedPath.h>
#include <cumulative_histogram/CumulativeHistogramImpl.h>
#include <cumulative_histogram/FullTreeView.h>
#include <cumulative_histogram/TreeView.h>

#include <algorithm>
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

// Container for efficient computation of prefix sums for a dynamic array of elements.
//
// The template type parameter T must satisfy the `Additive` concept above. There are additional
// requirements that cannot be expressed via C++ concepts:
// 1) Addition must be commutative, i.e. `a + b == b + a` for any a and b.
// 2) Addition must be associative, i.e. `(a + b) + c == a + (b + c)` for any a, b, c.
// 3) A default-constructed value-initialized object must have the same meaning as the identity element in
//    additive groups (i.e. "zero"): `T{} + x == x` for any x.
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
  // CumulativeHistogram splits the elements into buckets and builds an auxiliary binary tree for them
  // (the buckets are the leaves of the tree).
  // The asymptotic time complexities of the operations do not depend on the size of the bucket, but their
  // constant factors do.
  static constexpr std::size_t BucketSize = BucketSize<T>::value;
  static_assert(BucketSize >= 2, "BucketSize should not be less than 2.");

  using value_type = T;
  // TODO: declare as `typename std::vector<T>::size_type`.
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = const T&;
  using const_reference = const T&;
  using pointer = const T*;
  using const_pointer = const T*;
  using iterator = typename std::vector<T>::const_iterator;
  using const_iterator = typename std::vector<T>::const_iterator;
  using reverse_iterator = typename std::vector<T>::const_reverse_iterator;
  using const_reverse_iterator = typename std::vector<T>::const_reverse_iterator;

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
  // Time complexity: amortized O(1).
  void push_back();

  // Add an element to the end.
  // Time complexity: amortized O(1).
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

  // Swaps the contents of the current histogram with the given one.
  // Time complexity: O(1).
  void swap(CumulativeHistogram& other) noexcept(std::is_nothrow_swappable_v<std::vector<T>&>);

  // Sets the values of all elements to 0.
  // Time complexity: O(N).
  void setZero();

  // Sets the values of all elements to the specified value.
  // Time complexity: O(N).
  void fill(const T& value);

  // Access all elements.
  // Time complexity: O(1).
  constexpr const std::vector<T>& elements() const noexcept;

  // Access the specified element.
  // Throws std::out_of_range if k >= size().
  // Time complexity: O(1).
  const_reference element(size_type k) const;

  // Increment the specified element by the specified value.
  // \param k - 0-based index of the element to update.
  // \param value - value to add to the k-th element.
  // Throws std::out_of_range if k >= size().
  // Time complexity: O(log(N)).
  void increment(size_type k, const T& value);

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

  // Finds the first element k, for which prefixSum(k) is not less than the specified value.
  // The behavior is undefined if the prefix sums are not sorted in ascending order. For built-in
  // arithmetic types this can happen if any element is negative or if computing the total sum causes
  // overflow.
  // \return { begin()+k, prefixSum(k) }, where k is the first element for which !(prefixSum(k) < value),
  //         or { end(), T{} } if there is no such k.
  // Time complexity: O(log(N)).
  std::pair<const_iterator, T> lowerBound(const T& value) const;

  // Same as above, but uses the given comparison function instead of operator<.
  template<class Compare>
  std::pair<const_iterator, T> lowerBound(const T& value, Compare cmp) const;

  // Finds the first element k, for which prefixSum(k) is greater than the specified value.
  // The behavior is undefined if the prefix sums are not sorted in ascending order. For built-in
  // arithmetic types this can happen if any element is negative or if computing the total sum causes
  // overflow.
  // \return { begin()+k, prefixSum(k) }, where k is the first element for which value < prefixSum(k),
  //         or { end(), T{} } if there is no such k.
  // Time complexity: O(log(N)).
  std::pair<const_iterator, T> upperBound(const T& value) const;

  // Same as above, but uses the given comparison function instead of operator<.
  template<class Compare>
  std::pair<const_iterator, T> upperBound(const T& value, Compare cmp) const;

 private:
  // Returns an immutable TreeView for the currently effective tree.
  constexpr Detail_NS::TreeView<const T> getTreeView() const noexcept;

  // Returns an mutable TreeView for the currently effective tree.
  constexpr Detail_NS::TreeView<T> getMutableTreeView() noexcept;

  // Returns an immutable FullTreeView for the currently effective tree.
  constexpr Detail_NS::FullTreeView<const T> getFullTreeView() const noexcept;

  // Returns an mutable FullTreeView for the currently effective tree.
  constexpr Detail_NS::FullTreeView<T> getMutableFullTreeView() noexcept;

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

  // Path from the root of the currently effective tree to the leaf node that represents the last active bucket.
  // This path can be easily constructed by traversing the tree, but storing it in CumulativeHistogram has a benefit:
  // it allows to quickly (in O(1)) find the tree that needs to be extended after push_back(), and updating the path
  // itself after push_back() or pop_back() also takes O(1) time.
  // Without it, the time complexity of push_back() would've been O(logN) because we would have to traverse the tree
  // whenever a new node is added.
  Detail_NS::CompressedPath path_to_last_bucket_;
};

// Swaps the contents of the given histograms.
// Time complexity: O(1).
template<class T>
void swap(CumulativeHistogram<T>& lhs, CumulativeHistogram<T>& rhs)
noexcept(std::is_nothrow_swappable_v<std::vector<T>&>) {
  lhs.swap(rhs);
}

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
  // Computes the new capacity for a full tree that currently stores the specified number of elements.
  // \param capacity - capacity of the current tree.
  // \return the smallest capacity M of the new tree, such that
  //   M >= 2*capacity and the current tree is a subtree of the new tree.
  // Time complexity: O(1).
  constexpr std::size_t computeNewCapacityForFullTree(std::size_t capacity) noexcept {
    // Special case: if the tree is currently empty, then the new capacity is simply 2.
    if (capacity == 0) {
      return 2;
    }
    // Otherwise, there's at least 1 bucket already, so we should increase the number of buckets
    // from K to either 2K-1 or 2K (note that it can remain 1 if 2*Nmax elements also fit into a single bucket).
    // What is the smallest M such that
    //   M >= Nmax*2 AND
    //   (ceil(M/BucketSize) == 2*ceil(Nmax/BucketSize) OR
    //    ceil(M/BucketSize) == 2*ceil(Nmax/BucketSize) - 1)
    // ?
    // 1) Let's consider the case when Nmax < BucketSize.
    //   If 2*Nmax <= BucketSize, then the new number of buckets is also 1 = 2*1 - 1, so 2*Nmax is the answer.
    //   Otherwise, ceil(2*Nmax/BucketSize) = 2, because 2*Nmax < 2*BucketSize, so the new number of buckets
    //   is 2 = 2*1, and 2*Nmax is the answer.
    //   I.e. 2*Nmax is the answer in both cases.
    // 2) Let's consider the case when Nmax >= BucketSize.
    //   Let Nmax = a * BucketSize + b, where a > 0 and b < BucketSize.
    //   If b == 0, then 2*Nmax = 2*a*BucketSize is obviously the answer.
    //   Otherwise, ceil(2*Nmax/BucketSize) = ceil((2*a*BucketSize + 2*b)/BucketSize)
    //            = 2a + ceil(2*b/BucketSize)
    //     If 2*b < BucketSize, then the new number of buckets is 2a+1 = 2*(a+1) - 1;
    //     Otherwise, the new number of buckets is 2a+2 = 2*(a+1)
    //     Therefore, in both cases 2*Nmax is the answer.
    return capacity * 2;
  }

  // Computes the total sum of elements of a tree which is at its full capacity.
  // \param elements - elements to sum.
  // \param tree - auxiliary tree for `elements`.
  // The behavior is undefined if
  //   elements.empty() || elements.size() != tree.numBuckets() * bucket_size
  // \returns the total sum of elements from `elements`.
  // Time complexity: O(logN), where N = elements.size().
  template<class T>
  constexpr T sumElementsOfFullTree(std::span<const T> elements, FullTreeView<const T> tree, std::size_t bucket_size) {
    assert(!elements.empty());
    assert(elements.size() == tree.numBuckets() * bucket_size);
    T result{};
    while (!tree.empty()) {
      result += tree.root();
      tree.switchToRightChild();
    }
    // Add elements from the last bucket.
    const std::span<const T> last_bucket = elements.last(bucket_size);
    return std::accumulate(last_bucket.begin(), last_bucket.end(), std::move(result));
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
    assert(nodes.size() == countNodesInBucketizedTree(countBuckets(capacity, bucket_size)));
    const TreeViewData tree_data = getEffectiveTreeData(elements.size(), capacity, bucket_size);
    if (tree_data.num_nodes_at_level == 0) {
      return;
    }
    TreeView<T> tree{ nodes.subspan(tree_data.root_level, tree_data.num_nodes_at_level),
                      tree_data.num_buckets, tree_data.bucket_capacity_at_level };
    buildBucketizedTreeImpl(elements, tree, bucket_size);
  }

}  // namespace Detail_NS

template<Additive T>
constexpr Detail_NS::TreeView<const T> CumulativeHistogram<T>::getTreeView() const noexcept {
  const Detail_NS::TreeViewData tree_data = Detail_NS::getEffectiveTreeData(size(), capacity(), BucketSize);
  return Detail_NS::TreeView<const T> {
    std::span<const T> { nodes_.get() + tree_data.root_level, tree_data.num_nodes_at_level },
      tree_data.num_buckets, tree_data.bucket_capacity_at_level
  };
}

template<Additive T>
constexpr Detail_NS::TreeView<T> CumulativeHistogram<T>::getMutableTreeView() noexcept {
  const Detail_NS::TreeViewData tree_data = Detail_NS::getEffectiveTreeData(size(), capacity(), BucketSize);
  return Detail_NS::TreeView<T> {
    std::span<T> { nodes_.get() + tree_data.root_level, tree_data.num_nodes_at_level },
      tree_data.num_buckets, tree_data.bucket_capacity_at_level
  };
}

template<Additive T>
constexpr Detail_NS::FullTreeView<const T> CumulativeHistogram<T>::getFullTreeView() const noexcept {
  const std::size_t root_level = path_to_last_bucket_.rootLevel();
  const std::size_t bucket_capacity = path_to_last_bucket_.bucketCapacity();
  const std::size_t num_buckets_at_level = Detail_NS::countElementsInLeftmostSubtree(bucket_capacity, root_level);
  return Detail_NS::FullTreeView<const T>{ nodes_.get() + root_level, num_buckets_at_level };
}

template<Additive T>
constexpr Detail_NS::FullTreeView<T> CumulativeHistogram<T>::getMutableFullTreeView() noexcept {
  const std::size_t root_level = path_to_last_bucket_.rootLevel();
  const std::size_t bucket_capacity = path_to_last_bucket_.bucketCapacity();
  const std::size_t num_buckets_at_level = Detail_NS::countElementsInLeftmostSubtree(bucket_capacity, root_level);
  return Detail_NS::FullTreeView<T>{ nodes_.get() + root_level, num_buckets_at_level };
}

template<Additive T>
constexpr CumulativeHistogram<T>::CumulativeHistogram(const CumulativeHistogram& other):
  CumulativeHistogram(other.begin(), other.end())
{}

template<Additive T>
constexpr CumulativeHistogram<T>::CumulativeHistogram(CumulativeHistogram&& other) noexcept :
  elements_(std::move(other.elements_)),
  nodes_(std::move(other.nodes_)),
  capacity_(std::exchange(other.capacity_, static_cast<size_type>(0))),
  path_to_last_bucket_(std::move(other.path_to_last_bucket_))
{
}

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
  path_to_last_bucket_.build(Detail_NS::countBuckets(size(), BucketSize),
                             Detail_NS::countBuckets(capacity(), BucketSize));
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
  path_to_last_bucket_ = std::move(other.path_to_last_bucket_);
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
  path_to_last_bucket_.build(num_buckets, Detail_NS::countBuckets(capacity(), BucketSize));
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
  path_to_last_bucket_.build(num_buckets, Detail_NS::countBuckets(capacity(), BucketSize));
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
  return elements_.begin();
}

template<Additive T>
constexpr
typename CumulativeHistogram<T>::const_iterator CumulativeHistogram<T>::end() const noexcept {
  return elements_.end();
}

template<Additive T>
constexpr
typename CumulativeHistogram<T>::const_reverse_iterator CumulativeHistogram<T>::rbegin() const noexcept {
  return elements_.rbegin();
}

template<Additive T>
constexpr
typename CumulativeHistogram<T>::const_reverse_iterator CumulativeHistogram<T>::rend() const noexcept {
  return elements_.rend();
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

  // Depth of the currently effective tree. 0 means that the main tree is the currently effective tree.
  const size_type root_idx_old = path_to_last_bucket_.rootLevel();
  // The maximum number of buckets the currently effective tree can represent.
  const std::size_t bucket_capacity =
    Detail_NS::countElementsInLeftmostSubtree(path_to_last_bucket_.bucketCapacity(), root_idx_old);
  // The maximum number of buckets the new tree can represent.
  const std::size_t bucket_capacity_new = Detail_NS::countBuckets(num_elements, BucketSize);
  // Compute the number of nodes in the new tree.
  const std::size_t num_nodes_new = Detail_NS::countNodesInBucketizedTree(bucket_capacity_new);

  // Special case - if the new tree has 0 nodes, then that means that the original tree also has 0 nodes.
  if (num_nodes_new == 0) {
    elements_.reserve(num_elements);
    capacity_ = num_elements;
    path_to_last_bucket_.reserve(bucket_capacity_new);
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
  path_to_last_bucket_.reserve(bucket_capacity_new);
}

template<Additive T>
void CumulativeHistogram<T>::clear() noexcept {
  elements_.clear();
  path_to_last_bucket_.clear();
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
void CumulativeHistogram<T>::fill(const T& value) {
  std::fill(elements_.begin(), elements_.end(), value);
  const std::size_t num_buckets = Detail_NS::countBuckets(capacity_, BucketSize);
  const std::size_t num_nodes = Detail_NS::countNodesInBucketizedTree(num_buckets);
  const std::span<T> nodes{ nodes_.get(), num_nodes };
  // This can be optimized for types for which multiplication is defined and `x+x+x...+x == x*N`.
  // However, the time complexity will still be O(N), so whatever.
  Detail_NS::buildBucketizedTree<T>(elements_, nodes, capacity_, BucketSize);
}

template<Additive T>
void CumulativeHistogram<T>::push_back() {
  push_back(T{});
}

template<Additive T>
void CumulativeHistogram<T>::push_back(const T& value) {
  // Double the capacity if needed.
  if (size() == capacity()) {
    reserve(Detail_NS::computeNewCapacityForFullTree(capacity()));
  }
  // Check if adding an element will increase the number of buckets.
  if (size() % BucketSize != 0) {
    elements_.push_back(value);
    return;
  }
  // If we are adding the first bucket (i.e. the histogram was empty before this call), then
  // no nodes will be added because a tree representing a single bucket has 0 nodes.
  if (!empty()) {
    // Determine which node will need to be constructed after adding a new bucket.
    const Detail_NS::PathEntry subtree_to_extend = Detail_NS::findTreeToExtendAfterPushBack(path_to_last_bucket_);
    const std::size_t subtree_root_idx = path_to_last_bucket_.rootLevel() + subtree_to_extend.rootOffset();
    T* subtree_root = nodes_.get() + subtree_root_idx;
    T* new_node = subtree_root - 1;
    // Compute the sum of all elements in the effective right subtree.
    // This has O(logN) time complexity in the worst case, but, fortunately, the amortized time complexity is O(1).
    const std::size_t element_first = subtree_to_extend.bucketFirst() * BucketSize;
    const std::span<const T> subtree_elements = std::span<const T>(elements_).subspan(element_first);
    const Detail_NS::FullTreeView<const T> subtree(subtree_root, subtree_to_extend.numBuckets());
    // Construct the new node.
    *new_node = Detail_NS::sumElementsOfFullTree<T>(subtree_elements, subtree, BucketSize);
  }
  elements_.push_back(value);
  path_to_last_bucket_.pushBack();
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
  // Update the path to the last bucket if the number of buckets has changed.
  // The number of buckets changes only if there was K*BucketSize+1 elements before pop_back().
  if (size() % BucketSize == 0) {
    path_to_last_bucket_.popBack();
  }
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
    path_to_last_bucket_.build(Detail_NS::countBuckets(size(), BucketSize),
                               Detail_NS::countBuckets(capacity(), BucketSize));
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
void CumulativeHistogram<T>::swap(CumulativeHistogram& other)
noexcept(std::is_nothrow_swappable_v<std::vector<T>&>) {
  elements_.swap(other.elements_);
  std::swap(nodes_, other.nodes_);
  std::swap(capacity_, other.capacity_);
  path_to_last_bucket_.swap(other.path_to_last_bucket_);
}

template<Additive T>
constexpr const std::vector<T>& CumulativeHistogram<T>::elements() const noexcept {
  return elements_;
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
  // We are using FullTreeView here even though TreeView can skip inactive nodes in O(1)
  // time. The reason is that traversing FullTreeView is much faster - switching to the
  // left/right subtree only requires 3 or 4 instructions, but for TreeView it's much more.
  // This is essentially the constant factor of our O(logN) time complexity, and by
  // using FullTreeView we improve the average time complexity, whereas TreeView
  // optimizes the best-case time complexity at the cost of greater average time complexity.
  const size_type k_plus_one = k + 1;
  // Full view of the currently effective tree.
  Detail_NS::FullTreeView<T> tree = getMutableFullTreeView();
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
  // Full view of the currently effective tree.
  Detail_NS::FullTreeView<const T> tree = getFullTreeView();
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
  const std::size_t num_buckets = path_to_last_bucket_.numBuckets();
  T result {};
  // Full view of the currently effective tree.
  Detail_NS::FullTreeView<const T> tree = getFullTreeView();
  while (!tree.empty()) {
    // The tree represents buckets [first; last).
    // Its left subtree represents buckets [first; middle), and the right subtree represents [middle; last).
    // The root of the tree stores the sum of all elements from the buckets of the left subtree.
    // However, the root is only active if the right subtree is not empty.
    // The right subtree is empty if all elements are in the left subtree, i.e. if num_buckets == middle.
    if (num_buckets <= tree.pivot()) {
      tree.switchToLeftChild();
    }
    else {
      result += tree.root();
      tree.switchToRightChild();
    }
  }
  // Add elements from the last bucket.
  const size_type first = tree.bucketFirst() * BucketSize;
  return std::accumulate(elements_.begin() + first, elements_.end(), std::move(result));
}

template<Additive T>
std::pair<typename CumulativeHistogram<T>::const_iterator, T>
CumulativeHistogram<T>::lowerBound(const T& value) const {
  return lowerBound(value, std::less<T>{});
}

template<Additive T>
template<class Compare>
std::pair<typename CumulativeHistogram<T>::const_iterator, T>
CumulativeHistogram<T>::lowerBound(const T& value, Compare cmp) const {
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
CumulativeHistogram<T>::upperBound(const T& value) const {
  // Effectively implements `lhs <= rhs`, but only requires operator< to be defined for T.
  auto less_equal = [](const T& lhs, const T& rhs) { return !(rhs < lhs); };
  return lowerBound(value, less_equal);
}

template<Additive T>
template<class Compare>
std::pair<typename CumulativeHistogram<T>::const_iterator, T>
CumulativeHistogram<T>::upperBound(const T& value, Compare cmp) const {
  // Assuming that cmp(lhs, rhs) semantically means lhs < rhs, we can implement "less than or equal to"
  // comparison as !cmp(rhs, lhs).
  auto less_equal = [cmp](const T& lhs, const T& rhs) { return !cmp(rhs, lhs); };
  // lowerBound(value, less_equal) returns the first element k for which !less_equal(prefixSum(k), value),
  // i.e. the first element for which cmp(value, prefixSum(k)) == true, i.e. the first element for which
  // prefix sum is *greater* than value.
  return lowerBound(value, less_equal);
}

}  // namespace CumulativeHistogram_NS
