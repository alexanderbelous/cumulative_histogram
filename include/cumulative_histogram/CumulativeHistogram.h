#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace CumulativeHistogram_NS {

// A class for efficient computation of prefix sums for a dynamic array of elements.
template<class T>
class CumulativeHistogram {
 public:
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
  // Returns the index of the effective root node.
  // * nodes_[getRootIndex()] stores the sum of elements from the left subtree -
  //   but only if there is a left subtree (i.e. if there current tree has at least 1 node).
  // Time complexity: O(1).
  constexpr size_type getRootIndex() const noexcept;

  // Returns the maximum number of elements that can represented by the current tree.
  //   num_elements <= capacityCurrent() <= capacity_
  constexpr size_type capacityCurrent() const noexcept;

  // Returns the number of nodes in the current tree.
  constexpr size_type numNodesCurrent() const noexcept;

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
  // The number of nodes is always equal to Detail_NS::countNodesInTree(capacity_);
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

  // Returns the total number of nodes in the auxiliary tree for
  // CumulativeHistogram with the specified number of elements.
  // \param num_elements - the total number of elements represented by the tree.
  // Time complexity: O(1).
  constexpr std::size_t countNodesInTree(std::size_t num_elements) noexcept {
    // This number can be computed via a recurrent relation:
    //   f(0) = 0
    //   f(1) = 0
    //   f(2) = 0
    //   f(3) = 1
    //   f(4) = 1
    //   ...
    //   f(2N) = 2*f(N) + 1
    //   f(2N+1) = f(N+1) + f(N) + 1
    //
    // which forms the sequence https://oeis.org/A279521.
    if (num_elements <= 2) {
      return 0;
    }
    const std::size_t n = num_elements - 1;
    const std::size_t p2h = std::bit_floor(n);  // 2^h, where h = floor(log2(n))
    const std::size_t p2h_1 = p2h >> 1;         // 2^(h-1)
    return std::min(p2h - 1, n - p2h_1);
  }

  // Returns the height of the (full) tree for the specified number of elements.
  // \param num_elements - the total number of elements represented by the tree.
  // Time complexity: O(1).
  constexpr std::size_t heightOfFullTree(std::size_t num_elements) noexcept {
    return (num_elements > 1) ? floorLog2(num_elements - 1) : 0;
  }

  // Returns the number of nodes between the root (inclusive) and the rightmost
  // node (also inclusive) of the full tree for the specified number of elements.
  // \param num_elements - the total number of elements represented by the tree.
  // Returns 0 if the tree has no nodes.
  // Time complexity: O(1).
  constexpr std::size_t countNodesToRightmostNode(std::size_t num_elements) noexcept {
    const std::size_t num_nodes = countNodesInTree(num_elements);
    const std::size_t tree_height = heightOfFullTree(num_elements);
    const std::size_t num_nodes_in_full_binary_tree = (1 << tree_height) - 1;
    if (num_nodes < num_nodes_in_full_binary_tree) {
      return tree_height - 1;
    }
    return tree_height;
  }

  // Returns true if the rightmost node of the (full) tree for the specified number of elements
  // represents an even number of elements, false otherwise.
  // If the tree has no nodes, the function returns true if `num_elements` is even, or false if it's odd.
  // \param num_elements - the total number of elements represented by the tree.
  // Time complexity: O(1).
  constexpr bool rightmostNodeHasEvenNumberOfElements(std::size_t num_elements) noexcept {
    // Any leaf node represents either 3 or 4 elements, and its left branch always represents 2 elements.
    // Therefore, the rightmost node of the full tree represents an even number of elements (4) if
    // its right branch represents 2 elements, or an odd number of elements (3) if its right branch
    // represents just 1 element.
    //
    // Let f(N) be the number of elements in the right branch of the rightmost node
    // of a (full) tree for N elements, or simply N if that tree has no nodes.
    //
    // f(0) = 0    f(2) = 2    f(4) = 2    f(8) = 2     f(12) = 1
    // f(1) = 1    f(3) = 1    f(5) = 2    f(9) = 2     f(13) = 1
    //                         f(6) = 1    f(10) = 2    f(14) = 1
    //                         f(7) = 1    f(11) = 2    f(15) = 1
    //
    // This sequence is similar to https://oeis.org/A079944:
    //   0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1
    // which can be computed as
    //   h(N) = floor(log2(4*(N+2)/3)) - floor(log2(N+2))
    // for N=0,1,2,...
    //
    // f(N) = { 2 - h(N-2), if N >= 2;
    //        { 1,          if N == 1;
    //        { 0,          if N == 0.
    //
    // Note that floor(log2(x)) = floor(log2(floor(x)) for rational x >= 1, so
    // h(N - 2) can be computed as floorLog2(4 * num_elements / 3) - floorLog2(num_elements).
    if (num_elements <= 1) {
      return num_elements == 0;
    }
    // f(N) = 2 - h(N - 2) is even if h(N-2) == 0, and odd if h(N-2) = 1.
    return floorLog2(4 * num_elements / 3) == floorLog2(num_elements);
    // TODO: fix the overflow case.
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
    // The number of nodes in the leftmost subtree is computed as:
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
    //   All nodes in our tree represent at least 3 elements; therefore, any left subtree
    //   (which may have 0 nodes) represents at least 2 elements. Thus, computeRootIdxFast(1, Nmax)
    //   returns the same as computeRootIdxFast(2, Nmax).
    // * N=0 is an edge case because it will result into log2(-Nmax), which is undefined.
    //   In our case it means that we should return the deepest leftmost subtree, which is the
    //   same as calling computeRootIdxFast(2, Nmax).
    if (capacity < 2) return 0;
    if (num_elements < 2) num_elements = 2;
    // Note that ceil(log2(x)) = ceil(log2(ceil(x)).
    // TODO: fix the overflow case.
    const std::size_t ratio = (capacity + num_elements - 2) / (num_elements - 1);  // ceil(Nmax/(N-1))
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
    // Edge cases:
    // 1) A smaller tree cannot contain a larger tree.
    // 2) Our trees always have capacity >= 2. Trees of capacity == 1 are problematic - we forbid
    //    nodes with fewer than 2 elements, so a tree representing 1 element cannot be anyone's subtree.
    if (capacity < capacity_to_find || capacity_to_find < 2) {
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

  // High-level API for interacing with an implicit tree data structure.
  template<class T>
  class TreeView {
  public:
    using value_type = T;
    using reference = value_type&;

    // Expects that 0 < num_elements <= capacity.
    constexpr TreeView(std::span<value_type> nodes,
                       std::size_t num_elements,
                       std::size_t capacity) noexcept :
      TreeView(nodes, num_elements, 0, capacity - 1)
    {}

    // Converting constructor for an immutable TreeView from a mutable TreeView.
    template<class Enable = std::enable_if_t<std::is_const_v<T>>>
    constexpr TreeView(const TreeView<std::remove_const_t<T>>& tree) noexcept :
      TreeView(tree.nodes(), tree.capacity(), tree.elementFirst(), tree.elementTheoreticalLast())
    {}

    constexpr std::size_t elementFirst() const noexcept {
      return element_first_;
    }

    constexpr std::size_t elementTheoreticalLast() const noexcept {
      return element_theoretical_last_;
    }

    constexpr std::size_t capacity() const noexcept {
      return element_theoretical_last_ - element_first_ + 1;
    }

    constexpr bool empty() const noexcept {
      return nodes_.empty();
    }

    constexpr std::size_t numElements() const noexcept {
      return num_elements_;
    }

    constexpr std::span<value_type> nodes() const noexcept {
      return nodes_;
    }

    constexpr std::size_t pivot() const noexcept {
      return element_first_ + (element_theoretical_last_ - element_first_) / 2;
    }

    constexpr reference root() const {
      return nodes_.front();
    }

    constexpr TreeView leftChild() const noexcept {
      // The left subtree (if it exists) should always be at full capacity.
      const std::size_t capacity_left = (capacity() + 1) / 2;   // ceil(capacity() / 2)
      const std::size_t num_nodes_left = nodes_.size() / 2;     // ceil((nodes_.size() - 1) / 2)
      return TreeView(nodes_.subspan(1, num_nodes_left),
                      capacity_left, element_first_, pivot());
    }

    constexpr TreeView rightChild() const noexcept {
      const std::size_t num_nodes_left = nodes_.size() / 2;         // ceil((nodes_.size() - 1) / 2)
      const std::size_t capacity_total = capacity();
      const std::size_t capacity_left = (capacity_total + 1) / 2;   // ceil(capacity_total / 2)
      const std::size_t capacity_right = capacity_total / 2;        // floor(capacity_total / 2)
      const std::size_t element_pivot = pivot();
      const std::size_t num_elements_right = num_elements_ - capacity_left;
      // Find the deepest leftmost subtree of the immediate right subtree that represents all
      // elements [element_pivot + 1; element_pivot + num_elements_right].
      const std::size_t level = findDeepestNodeForElements(num_elements_right, capacity_right);
      const std::size_t capacity_at_level = countElementsInLeftmostSubtree(capacity_right, level);
      const std::size_t num_nodes_at_level = countNodesInTree(capacity_at_level);
      // Skip the 0th node because it's the root.
      // Skip the next `num_nodes_left` because they belong to the left subtree.
      // Skip the next `level` nodes because those are nodes between the root of our
      //      "effective" right subtree and the root of the current tree.
      const std::span<value_type> nodes_at_level = nodes_.subspan(1 + num_nodes_left + level, num_nodes_at_level);
      return TreeView(nodes_at_level,
                      num_elements_right, element_pivot + 1, element_pivot + capacity_at_level);
    }

  private:
    constexpr TreeView(std::span<value_type> nodes,
                       std::size_t num_elements,
                       std::size_t element_first,
                       std::size_t element_theoretical_last) noexcept :
      nodes_(nodes),
      num_elements_(num_elements),
      element_first_(element_first),
      element_theoretical_last_(element_theoretical_last)
    {}

    std::span<value_type> nodes_;
    // The number of real elements represented by the tree.
    std::size_t num_elements_;
    // Indices of the first and last (inclusive) elements that *can* be represented by the tree.
    std::size_t element_first_;
    std::size_t element_theoretical_last_;
  };

  // Computes the sum of all elements of the given nodeless tree.
  // Such trees always represent either 1 or 2 elements, so there's no need for a loop.
  // Time complexity: O(1).
  template<class T>
  constexpr T sumElementsOfEmptyTree(std::span<const T> elements, const TreeView<const T>& tree) {
    const std::size_t num_elements = tree.numElements();
    assert(0 < num_elements && num_elements <= 2);
    const std::size_t first = tree.elementFirst();
    if (num_elements == 1) {
      return elements[first];
    }
    return elements[first] + elements[first + 1];
  }

  // This function is intended to be called for trees at their full capacity.
  // Computes the total sum of elements of a tree which is at its full capacity.
  // \param elements - elements represented by the tree.
  // \param nodes - node of the tree.
  // The behavior is undefined if
  //   elements.empty() || nodes.size() != countNodesInTree(elements.size())
  // \returns the total sum of elements from `elements`.
  // Time complexity: O(logN), where N = elements.size().
  template<class T>
  constexpr T sumElementsOfFullTree(std::span<const T> elements, std::span<const T> nodes) {
    assert(!elements.empty());
    assert(nodes.size() == countNodesInTree(elements.size()));
    const std::size_t num_elements = elements.size();
    const std::size_t path_length_to_rightmost_node = countNodesToRightmostNode(num_elements);
    const T* node = nodes.data();
    std::size_t num_nodes = nodes.size();
    T result{};
    // The loop is equivalent to `while (num_nodes != 0)`, but we can count the
    // number of iterations in O(1).
    for (std::size_t depth = 0; depth < path_length_to_rightmost_node; ++depth) {
      result += *node;
      // Switch to the right subtree.
      const std::size_t num_nodes_left = (num_nodes >> 1);  // ceil((num_nodes - 1) / 2);
      node += (1 + num_nodes_left);
      num_nodes = (num_nodes - 1) >> 1;  // floor((num_nodes - 1) / 2);
    }
    // The rightmost subtree represents either 3 or 4 elements. The root of that subtree
    // (which is its only node) stores the sum of the first 2 of these elements, so we need
    // to add the values of elements[N-1] and, maybe, elements[N-2].
    if (rightmostNodeHasEvenNumberOfElements(num_elements)) {
      result += elements[num_elements - 2];
    }
    result += elements[num_elements - 1];
    return result;
  }

  // Builds the tree for the given elements.
  // Expects that:
  // 1) 0 <= elements.size() <= capacity
  // 2) nodes.size() == countNodesInTree(capacity)
  // Time complexity: O(N), where N = elements.size().
  // TODO: change the API so that `nodes` is only required to have enough nodes to represent all elements.
  //       Or just pass const std::vector& and let buildTree() decide the optimal structure.
  template<class T>
  void buildTree(std::span<const T> elements, std::span<T> nodes, std::size_t capacity) {
    if (elements.empty()) {
      return;
    }
    const std::size_t level = findDeepestNodeForElements(elements.size(), capacity);
    const std::size_t capacity_at_level = countElementsInLeftmostSubtree(capacity, level);
    const std::size_t num_nodes_at_level = countNodesInTree(capacity_at_level);
    const std::span<T> nodes_at_level = nodes.subspan(level, num_nodes_at_level);
    TreeView<T> tree(nodes_at_level, elements.size(), capacity_at_level);
    buildTreeImpl(elements, tree);
  }

  // Initializes the nodes of the specified tree according to the values of the given elements.
  // \param elements - values of elements for which we want to track prefix sums.
  // \param tree - tree for some or all elements from `elements`.
  // \returns the total sum of elements represented by `tree`.
  template<class T>
  T buildTreeImpl(std::span<const T> elements, const TreeView<T>& tree) {
    // Iterative version
    /*
    if (tree.empty())
    {
      return sumElementsOfEmptyTree<T>(elements, tree);
    }
    struct StackVariables {
      TreeView<true> tree;
      T* sum_dest = nullptr;
      bool done = false;
    };
    std::vector<StackVariables> stack;
    // Reserve memory for all nodes.
    stack.reserve(countNodesInTree(elements.size()));
    T total_sum {};
    stack.push_back(StackVariables{ .tree = tree, .sum_dest = &total_sum, .done = false });
    while (!stack.empty()) {
      StackVariables& vars = stack.back();
      // vars.tree.empty() is always false.
      // TODO: don't *modify* nodes that represent elements [a; b], where b>elements.size()-1.
      // Rationale: these nodes will never be accessed during increment() or prefixSum().
      if (vars.done) {
        // Add the sum of elements from the left subtree to OUR return value,
        // which already contains the sum of elements from the right subtree.
        *(vars.sum_dest) += vars.tree.root();
        stack.pop_back();
      } else {
        vars.done = true;
        // Zero-initialize the root.
        vars.tree.root() = T {};
        // The sum of elements from the left subtree should be added to the root of the current tree.
        const TreeView<true> left_child = vars.tree.leftChild();
        if (left_child.empty()) {
          vars.tree.root() += sumElementsOfEmptyTree<T>(elements, left_child);
        } else {
          // Schedule a call to build the left subtree.
          stack.push_back(StackVariables{ .tree = left_child, .sum_dest = &vars.tree.root(), .done = false });
        }
        // The sum of elements from the right subtree should be added to OUR return value.
        const TreeView<true> right_child = vars.tree.rightChild();
        if (right_child.empty()) {
          *(vars.sum_dest) += sumElementsOfEmptyTree<T>(elements, right_child);
        } else {
          // Schedule a call to build the right subtree.
          stack.push_back(StackVariables{ .tree = right_child, .sum_dest = vars.sum_dest, .done = false });
        }
      }
    }
    return total_sum;
    */

    // Tail recursion optimization
    /*
    TreeView<true> t = tree;
    T total_sum {};
    while (!t.empty()) {
      const T total_sum_left = buildTreeImpl(elements, t.leftChild());
      t.root() = total_sum_left;
      total_sum += total_sum_left;
      t = t.rightChild();
    }
    return total_sum + sumElementsOfEmptyTree<T>(elements, t);
    */

    // Two recursive calls
    // Lol, surprisingly, this is the most efficient version.
    if (tree.empty()) {
      return sumElementsOfEmptyTree<T>(elements, tree);
    }
    const T total_sum_left = buildTreeImpl(elements, tree.leftChild());
    tree.root() = total_sum_left;
    return total_sum_left + buildTreeImpl(elements, tree.rightChild());
  }

}  // namespace Detail_NS

template<class T>
constexpr
typename CumulativeHistogram<T>::size_type
CumulativeHistogram<T>::getRootIndex() const noexcept {
  return Detail_NS::findDeepestNodeForElements(size(), capacity());
}

template<class T>
constexpr
typename CumulativeHistogram<T>::size_type
CumulativeHistogram<T>::capacityCurrent() const noexcept {
  // Determine the depth of the current root node in the "full" tree.
  // Note that there aren't actually any nodes if capacity_ < 3. In that case
  // level will be 0, and this function will simply return capacity_.
  const std::size_t level = Detail_NS::findDeepestNodeForElements(size(), capacity());
  return Detail_NS::countElementsInLeftmostSubtree(capacity(), level);
}

template<class T>
constexpr std::size_t CumulativeHistogram<T>::numNodesCurrent() const noexcept {
  return Detail_NS::countNodesInTree(capacityCurrent());
}

template<class T>
constexpr CumulativeHistogram<T>::CumulativeHistogram(const CumulativeHistogram& other):
  CumulativeHistogram(other.begin(), other.end())
{}

template<class T>
constexpr CumulativeHistogram<T>::CumulativeHistogram(CumulativeHistogram&& other) noexcept :
  elements_(std::move(other.elements_)),
  nodes_(std::move(other.nodes_)),
  capacity_(std::exchange(other.capacity_, static_cast<size_type>(0)))
{}

template<class T>
constexpr CumulativeHistogram<T>& CumulativeHistogram<T>::operator=(const CumulativeHistogram& other)
{
  if (capacity_ < other.size()) {
    // Delegate to copy constructor and move assignment operator.
    return *this = CumulativeHistogram(other);
  }
  // Our capacity is sufficient to store all elements from `other`, so no memory allocation is needed.
  elements_.clear();
  elements_.insert(elements_.end(), other.begin(), other.end());
  const size_type num_nodes = Detail_NS::countNodesInTree(capacity_);
  const std::span<T> nodes{ nodes_.get(), num_nodes };
  Detail_NS::buildTree<T>(elements_, nodes, capacity_);
  return *this;
}

template<class T>
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

template<class T>
CumulativeHistogram<T>::CumulativeHistogram(size_type num_elements):
  elements_(num_elements),
  capacity_(num_elements)
{
  if (capacity_ != 0) {
    const size_type num_nodes = Detail_NS::countNodesInTree(capacity_);
    // Zero-initialize the nodes.
    nodes_ = std::make_unique<T[]>(num_nodes);
  }
}

template<class T>
CumulativeHistogram<T>::CumulativeHistogram(size_type num_elements, const T& value):
  CumulativeHistogram(std::vector<T>(num_elements, value))
{
}

template<class T>
CumulativeHistogram<T>::CumulativeHistogram(std::vector<T>&& elements):
  elements_(std::move(elements)),
  capacity_(elements_.size())
{
  if (elements_.empty()) {
    return;
  }
  // TODO: only construct nodes that are needed to represent the current level.
  const size_type num_nodes = Detail_NS::countNodesInTree(capacity_);
  // Default-initialize the nodes - there's no need to zero-initialize them.
  nodes_ = std::make_unique_for_overwrite<T[]>(num_nodes);
  const std::span<T> nodes{ nodes_.get(), num_nodes };
  Detail_NS::buildTree<T>(elements_, nodes, capacity());
}

template<class T>
template<std::input_iterator Iter>
CumulativeHistogram<T>::CumulativeHistogram(Iter first, Iter last):
  CumulativeHistogram(std::vector<T>(first, last))
{
}

template<class T>
constexpr
typename CumulativeHistogram<T>::const_iterator CumulativeHistogram<T>::begin() const noexcept {
  return elements_.data();
}

template<class T>
constexpr
typename CumulativeHistogram<T>::const_iterator CumulativeHistogram<T>::end() const noexcept {
  return elements_.data() + size();
}

template<class T>
constexpr
typename CumulativeHistogram<T>::const_reverse_iterator CumulativeHistogram<T>::rbegin() const noexcept {
  return std::make_reverse_iterator(end());
}

template<class T>
constexpr
typename CumulativeHistogram<T>::const_reverse_iterator CumulativeHistogram<T>::rend() const noexcept {
  return std::make_reverse_iterator(begin());
}

template<class T>
constexpr bool CumulativeHistogram<T>::empty() const noexcept {
  return elements_.empty();
}

template<class T>
constexpr
typename CumulativeHistogram<T>::size_type
CumulativeHistogram<T>::size() const noexcept {
  return elements_.size();
}

template<class T>
constexpr
typename CumulativeHistogram<T>::size_type
CumulativeHistogram<T>::capacity() const noexcept {
  return capacity_;
}

template<class T>
void CumulativeHistogram<T>::reserve(size_type num_elements) {
  if (num_elements <= capacity()) {
    return;
  }
  // Compute the minimum number of nodes in the tree that can represent `num_elements` elements.
  const std::size_t num_nodes_new = Detail_NS::countNodesInTree(num_elements);
  // Allocate memory for the new tree.
  // TODO: only construct the nodes that are needed to represent the current level.
  std::unique_ptr<T[]> new_nodes = std::make_unique_for_overwrite<T[]>(num_nodes_new);
  const std::span<T> new_nodes_span{ new_nodes.get(), num_nodes_new };
  // Get the capacity of the currently effective tree.
  const size_type capacity_current = capacityCurrent();
  // Check the special case when the tree for num_elements has our current tree as a subtree.
  // In that case there's no need to rebuild the tree - we can just copy our current one.
  const std::size_t level_for_the_original =
    Detail_NS::findLeftmostSubtreeWithExactCapacity(capacity_current, num_elements);
  // Construct the new tree.
  if (level_for_the_original == static_cast<std::size_t>(-1)) {
    // The old tree is not a subtree of the new tree, so we have to build the new one from scratch.
    Detail_NS::buildTree<T>(elements_, new_nodes_span, num_elements);
    // Reserve new data for elements.
    elements_.reserve(num_elements);
  } else {
    // Just copy the current tree as a subtree of the new one.
    const size_type root_idx_old = getRootIndex();
    const size_type num_nodes_old = Detail_NS::countNodesInTree(capacity_current);
    const std::span<T> effective_nodes_old { nodes_.get() + root_idx_old, num_nodes_old };
    const std::span<T> effective_nodes_new = new_nodes_span.subspan(level_for_the_original, num_nodes_old);
    // Basic exception guarantee: we only move the nodes if T's move assignment is noexcept;
    // otherwise, we copy them, so that even if an exception is thrown during copying, this class
    // will remain in a valid state.
    // TODO: replace the condition with std::std::is_nothrow_constructible_v after implementing
    // lifetimes for nodes.
    if constexpr (std::is_nothrow_move_assignable_v<T>) {
      // Memory is reserved before we move the nodes to ensure strong exception guarantee:
      // std::vector::reserve() may throw an exception, but std::copy_n() cannot.
      elements_.reserve(num_elements);
      std::copy_n(std::make_move_iterator(effective_nodes_old.begin()), effective_nodes_old.size(),
                  effective_nodes_new.begin());
    } else {
      std::copy_n(effective_nodes_old.cbegin(), effective_nodes_old.size(),
                  effective_nodes_new.begin());
      // std::vector::reserve() happens after we copy the tree to ensure strong exception guarantee:
      // copying the tree doesn't modify *this, and std::vector::reserve() itself won't modify
      // `elements_` if an exception is thrown.
      elements_.reserve(num_elements);
    }
  }
  // Replace old data with new data.
  nodes_ = std::move(new_nodes);
  capacity_ = num_elements;
}

template<class T>
void CumulativeHistogram<T>::clear() noexcept {
  elements_.clear();
  // TODO: destroy the nodes.
}

template<class T>
void CumulativeHistogram<T>::setZero() {
  std::fill(elements_.begin(), elements_.end(), T{});
  const std::span<T> nodes_current{ nodes_.get() + getRootIndex(), numNodesCurrent() };
  // TODO: only fill the nodes that are actually used.
  std::fill(nodes_current.begin(), nodes_current.end(), T{});
}

template<class T>
void CumulativeHistogram<T>::push_back() {
  push_back(T{});
}

template<class T>
void CumulativeHistogram<T>::push_back(const T& value) {
  // Double the capacity if needed.
  if (size() == capacity()) {
    const size_type capacity_new = (capacity() == 0) ? 2 : (capacity() * 2);
    reserve(capacity_new);
  }

  // There are 2 possibilities - either we can simply add the new element without constructing
  // any new nodes, or we need to extend some subtree by constructing a new root.
  const std::span<T> nodes = std::span<T>{ nodes_.get(), Detail_NS::countNodesInTree(capacity()) };
  const std::span<const T> nodes_effective = nodes.subspan(getRootIndex(), numNodesCurrent());
  Detail_NS::TreeView<const T> tree { nodes_effective, size(), capacityCurrent() };
  // TODO: get rid of this loop. Traversing the tree has O(logN) time complexity, but it can be avoided
  // if we store the path to the deepest rightmost subtree. In that case updating the path can be done in O(1).
  while (tree.numElements() != tree.capacity()) {
    // If the tree has no nodes (which means it can represent at most 2 elements) and is not at full capacity,
    // we can simply add the new element without constructing any new nodes.
    if (tree.empty()) {
      elements_.push_back(value);
      return;
    }
    // Otherwise, the left subtree must be full, so we switch to the effective right subtree.
    tree = tree.rightChild();
  }
  // If we are here, then we have found some non-empty subtree (maybe the main tree) that is at full capacity.
  // This subtree cannot be the *immediate* right subtree of some existing tree, because that would mean that
  // this tree is also at full capacity, which contradicts the loop condition - we were searching for the topmost
  // full subtree.
  // Hence, this subtree is NOT the *immediate* right subtree, i.e. it's the left subtree (at some depth K) of the
  // *immediate* right subtree. We will construct a new root, whose left subtree will be the tree we've found, and
  // the right subtree will store the element that we are adding.
  // TreeView doesn't store the index of the root node, but it's easy to compute via pointer arithmetic.
  const std::size_t tmp_root_idx = tree.nodes().data() - nodes.data();
  const std::size_t root_idx_new = tmp_root_idx - 1;
  // Compute the sum of all elements in the effective right subtree.
  // This has O(logN) time complexity in the worst case, but, fortunately, the amortized time complexity is O(1).
  const std::span<const T> tmp_elements = std::span<const T>(elements_).subspan(tree.elementFirst(), tree.numElements());
  // Construct the new node.
  nodes[root_idx_new] = Detail_NS::sumElementsOfFullTree(tmp_elements, tree.nodes());
  // The new element is added to the right subtree of the newly constructed tree. This subtree doesn't
  // have any nodes yet, because it only represents 1 element.
  elements_.push_back(value);
}

template<class T>
void CumulativeHistogram<T>::pop_back() {
  if (empty()) {
    throw std::logic_error("CumulativeHistogram::pop_back(): there are no elements left to remove.");
  }
  // TODO: find the deepest rightmost subtree. If we are removing the only element from that subtree,
  // then we should destroy its *effective* root node.
  // Currently, though, we don't construct/destroy nodes, so whatever.
  elements_.pop_back();
}

template<class T>
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

template<class T>
constexpr std::span<const T> CumulativeHistogram<T>::elements() const noexcept {
  return std::span<const T>{elements_.data(), size()};
}

template<class T>
typename CumulativeHistogram<T>::const_reference
CumulativeHistogram<T>::element(size_type k) const {
  return elements_.at(k);
}

template<class T>
void CumulativeHistogram<T>::increment(size_type k, const T& value) {
  if (k >= size()) {
    throw std::out_of_range("CumulativeHistogram::increment(): k is out of range.");
  }
  // Tree representing the elements [0; N).
  const size_type root_idx = getRootIndex();
  const std::span<T> nodes = std::span<T>{ nodes_.get() + root_idx, numNodesCurrent() };
  Detail_NS::TreeView<T> tree(nodes, size(), capacityCurrent());
  while (!tree.empty()) {
    // Check whether the element k is in the left or the right branch.
    if (k <= tree.pivot()) {
      // The root stores the sum of all elements in the left subtree, so we need to increment it.
      tree.root() += value;
      tree = tree.leftChild();
    } else {
      tree = tree.rightChild();
    }
  }
  // Update the element itself.
  elements_[k] += value;
}

template<class T>
T CumulativeHistogram<T>::prefixSum(size_type k) const {
  if (k >= size()) {
    throw std::out_of_range("CumulativeHistogram::prefixSum(): k is out of range.");
  }
  // Special case for the total sum.
  if (k == size() - 1) {
    return totalSum();
  }
  T result {};
  // Tree representing the elements [0; N).
  const size_type root_idx = getRootIndex();
  const std::span<const T> nodes = std::span<const T>{ nodes_.get() + root_idx, numNodesCurrent() };
  Detail_NS::TreeView<const T> tree(nodes, size(), capacityCurrent());
  while (!tree.empty()) {
    // The root of the tree stores the sum of all elements [first; middle].
    const std::size_t middle = tree.pivot();
    if (k < middle) {
      tree = tree.leftChild();
    } else if (k == middle) {
      return result + tree.root();
    } else {
      result += tree.root();
      tree = tree.rightChild();
    }
  }
  // If we are here, then the value of x[k] itself hasn't been added through any node in the tree.
  result += elements_[k];
  return result;
}

template<class T>
T CumulativeHistogram<T>::totalSum() const {
  if (empty()) {
    throw std::logic_error("CumulativeHistogram::totalSum(): the histogram is empty.");
  }
  T result {};
  const size_type root_idx = getRootIndex();
  const std::span<const T> nodes = std::span<const T>{ nodes_.get() + root_idx, numNodesCurrent() };
  Detail_NS::TreeView<const T> tree(nodes, size(), capacityCurrent());
  while (!tree.empty()) {
    result += tree.root();
    tree = tree.rightChild();
  }
  // Add values of existing elements from the last tree.
  result += elements_[tree.elementFirst()];
  if (tree.numElements() > 1) {
    result += elements_[tree.elementTheoreticalLast()];
  }
  return result;
}

template<class T>
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
  const size_type root_idx = getRootIndex();
  const std::span<const T> nodes = std::span<const T>{ nodes_.get() + root_idx, numNodesCurrent() };
  Detail_NS::TreeView<const T> tree(nodes, size(), capacityCurrent());
  while (!tree.empty()) {
    // The root of the tree stores the sum of all elements [k_lower; middle].
    // Sum of elements [0; middle]
    T prefix_sum_middle = prefix_sum_before_lower + tree.root();
    if (cmp(prefix_sum_middle, value)) {
      // OK, we don't need to check the left tree, because prefixSum(i) < value for i in [0; middle].
      // k_lower = middle + 1;
      prefix_sum_before_lower = std::move(prefix_sum_middle);
      tree = tree.rightChild();
    } else {
      // No need to check the right tree because prefixSum(i) >= value for i in [middle; N).
      // Note that it's still possible that middle is the element we're looking for.
      // k_upper = middle;
      prefix_sum_upper = std::move(prefix_sum_middle);
      tree = tree.leftChild();
    }
  }
  // We know that cmp(prefixSum(i), value) == true for all i < k_lower
  const std::size_t k_lower = tree.elementFirst();
  // Compute prefixSum(k_lower).
  prefix_sum_before_lower += elements_[k_lower];
  if (!cmp(prefix_sum_before_lower, value)) {
    // OK, k_lower is the answer.
    return { begin() + k_lower, prefix_sum_before_lower };
  }
  // We know that cmp(prefixSum(i), value) == false for all i > k_upper (if there is such i).
  const std::size_t k_upper = k_lower + tree.numElements() - 1;
  // If k_upper is the last element, then `prefix_sum_upper` hasn't been initialized.
  if (k_upper == size() - 1) {
    // If k_lower == k_upper, then cmp(totalSum(), value) == cmp(prefixSum(k_lower), value) == true.
    if (k_lower == k_upper) {
      return { end(), T{} };
    }
    prefix_sum_upper = prefix_sum_before_lower + elements_[k_upper];
    // If cmp(totalSum(), value) == true, then there is no such index k that cmp(prefixSum(k), value) == false.
    if (cmp(prefix_sum_upper, value)) {
      return { end(), T{} };
    }
  }
  // OK, cmp(prefixSum(k), value) == false, so k_upper is the answer.
  return { begin() + k_upper, prefix_sum_upper };
}

template<class T>
std::pair<typename CumulativeHistogram<T>::const_iterator, T>
CumulativeHistogram<T>::lowerBound(const T& value) const {
  return lowerBoundImpl(value, std::less<T>{});
}

template<class T>
std::pair<typename CumulativeHistogram<T>::const_iterator, T>
CumulativeHistogram<T>::upperBound(const T& value) const {
  // Effectively implements `lhs <= rhs`, but only requires operator< to be defined for T.
  auto less_equal = [](const T& lhs, const T& rhs) { return !(rhs < lhs); };
  return lowerBoundImpl(value, less_equal);
}

}  // namespace CumulativeHistogram_NS
