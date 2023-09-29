#pragma once

#include <bit>
#include <iterator>
#include <stdexcept>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace CumulativeHistogram_NS {

// This class allows to efficiently compute prefix sums for a dynamic array of elements
// by offering a compromise between 2 naive solutions:
// +----------------------+-----------------+-------------+--------------+-------+
// | Solution             | updateElement() | prefixSum() | lowerBound() | Space |
// +----------------------+-----------------+-------------+--------------+-------+
// | Array of elements    | O(1)            | O(N)        | O(N)         | O(N)  |
// +----------------------+-----------------+-------------+--------------+-------+
// | Array of prefix sums | O(N)            | O(1)        | O(logN)      | O(N)  |
// +----------------------+-----------------+-------------+--------------+-------+
// | CumulativeHistogram  | O(logN)         | O(logN)     | O(logN)      | O(N)  |
// +----------------------+-----------------+-------------+--------------+-------+
//
// Unlike a Fenwick tree, this class allows adding and removing elements.
// The memory overhead is approximately 50% (i.e. the class stores roughly N/2 extra counters).
template<class T>
class CumulativeHistogram {
 public:
  using value_type = T;
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

  // Constructs a cumulative histogram for N zero-initialized elements.
  // Time complexity: O(N).
  explicit CumulativeHistogram(size_type num_elements);

  // Constructs a cumulative histogram for N elements, initializing them with the specified value.
  // Time complexity: O(N).
  explicit CumulativeHistogram(size_type num_elements, const T& value);

  // Constructs a cumulative histogram for the specified elements.
  // Time complexity: O(N).
  explicit CumulativeHistogram(std::vector<T>&& elements);

  // Constructs a cumulative histogram for the specified elements.
  // This overload only participates in overload resolution if Iter satisfies
  // std::input_iterator concep, to avoid ambiguity with CumulativeHistogram(std::size_t, T).
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
  // Time complexity: O(N), where N is the current number of elements.
  void reserve(size_type num_elements);

  // Erases all elements.
  // The capacity remains unchanged.
  // Time complexity: O(N).
  // Note: time complexity is linear because we need to call the destructor for each element.
  // For arithmetic types we could skip that, but T can be a class/struct (e.g., std::complex or some BigInt).
  // TODO: consider customizing the behavior for trivial types.
  void clear() noexcept;

  // Add a zero-initialized element to the end.
  // Time complexity: amortized O(1).
  void push_back();

  // Add an element to the end.
  // Time complexity: amortized O(logN).
  // TODO: make it amortized O(1).
  void push_back(const T& value);

  // Removes the last element.
  // Throws std::logic_error if this->empty().
  // Time complexity: O(logN).
  //
  // TODO:
  //   AFAIU, it should be possible to make it O(1) at the cost of
  // making push_back() a bit slower (though still having amortized O(logN) time complexity).
  //   When the rightmost element is removed, only the total sum actually needs to be updated.
  // Yes, other nodes are affected, but the affected nodes will not be accesed anymore
  // (the node is affected if it contains x[i]; if we remove x[i] then no valid call to
  // prefixSum() will actually add values from the nodes that contain x[i]).
  //   However, this means that these nodes will store garbage data; if we call push_back()
  // afterwards, we'll need to ensure that it zero-initializes these nodes before incrementing
  // them. This can still be done in O(logN) - e.g., push_back() can initialize the
  // nodes that contain the new element, and there's at most O(log) of them.
  //   Then again, we might want to do this in push_back() anyway for 2 reasons:
  //     1. To avoid zeroing out newly allocated memory. reserve(M) will have O(M) time coplexity
  //        if we zero-initialize, but only O(N) if we don't.
  //     2. To reduce the rounding errors when T is a floating-point type.
  //
  // TODO:
  //   Should I call ~T() at least for the element that has been removed?
  //   ~T() will likely be trivial, but doesn't have to be - e.g., one might want to instantiate
  //   CumulativeHistogram for some BigInt class that uses heap storage.
  //   Pros: It will reduce memory usage for the scenarios like BigInt.
  //   Cons: I'd rather avoid tracking the lifetimes of individual counters.
  void pop_back();

  // Changes the number of elements stored.
  // \param num_elements - the new number of elements in the histogram.
  // Time complexity: O(|N' - N|), if capacity() >= num_elements,
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
  constexpr T totalSum() const;

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

  template<bool Mutable>
  class TreeView;

  template<bool Mutable>
  class TreeViewAdvanced;

  // Builds the tree for the given elements.
  // Expects that:
  // 1) 0 <= elements.size() <= capacity
  // 2) nodes.size() == 1 + countNodesInTree(capacity)
  // Time complexity: O(elements.size())
  // TODO: change the API so that `nodes` is only required to have enough nodes to represent all elements.
  //       Or just pass const std::vector& and let buildTree() decide the optimal structure.
  static void buildTree(std::span<const T> elements, std::span<T> nodes, std::size_t capacity);

  // Initializes the nodes of the specified TreeView according to the values of the given elements.
  // \param elements - values of elements for which we want to track prefix sums.
  // \param tree - TreeView for some or all elements from `elements`.
  // \returns the total sum of elements represented by `tree`.
  static T buildTreeImpl(std::span<const T> elements, const TreeView<true>& tree);

  // This function is intended to be called for empty trees (trees without nodes).
  // Such trees always represent either 1 or 2 elements, so there's no need for a loop.
  static constexpr T sumElementsOfEmptyTree(std::span<const T> elements, const TreeView<false>& tree);

  // This function is intended to be called for trees at their full capacity.
  static constexpr T sumElementsOfFullTree(std::span<const T> elements, TreeView<false> tree);

  // Values of elements in the histogram.
  std::vector<T> elements_;
  // Nodes of the tree.
  std::vector<T> nodes_;
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
// +------------------------------------------------------------------------------------------+
// |  s0  |  s1  |  s2  |  s3   |  s4   |  s5  |  s6   |  s7   |  s8   |  s9      |  s10      |
// +------------------------------------------------------------------------------------------+
// |  x0  |  n2  |  n1  | n1+x3 | n1+n3 |  n0  | n0+x6 | n0+n5 | n0+n4 | n0+n4+x9 | total_sum |
// +------------------------------------------------------------------------------------------+
//
// In order to support efficient insertion, the class grows in the same way that std::vector does:
// if size() == capacity(), then the next call to push_back() will allocate memory for a histogram
// capable of storing 2N elements.
//
// The data is stored like this:
// * N = the number of elements.
// * Nmax = current capacity.
// * M = countNodesInTree(Nmax)
// +------+------+------+------+------+------------+------+--------+--------+--------+------+----------+
// |   0  |   1  |   2  | .... |  N-1 | .......... | Nmax | Nmax+1 | Nmax+2 | Nmax+3 | ...  | Nmax+M-1 |
// +------+------+------+------+------+------------+------+--------+--------+--------+------+----------+
// |  x0  |  x1  |  x2  | .... | xN-1 | ...free... |  n0  |   n1   |   n2   |   n3   | .... |   nM-1   |
// +------+------+------+------+------+------------+------+--------+--------+--------+------+----------+
// In practice, the root is not necessarily at n0: if the current number of elements is <= ceil(N/2),
// then the right subtree is empty, and n0 is the same as total_sum, so we may as well pretend that
// the left subtree *is* the tree, and that the root is at n1. This is applied recursively until
// the right subtree is not empty. This way we ensure that the time complexity of the operations
// is actually O(logN), and not O(logNmax).
template<class T>
template<bool Mutable>
class CumulativeHistogram<T>::TreeView {
public:
  using value_type = std::conditional_t<Mutable, T, const T>;
  using reference = value_type&;

  constexpr TreeView(std::span<value_type> nodes,
                     std::size_t element_first,
                     std::size_t element_last) noexcept :
    nodes_(nodes),
    element_first_(element_first),
    element_last_(element_last)
  {}

  // Converting constructor for an immutable TreeView from a mutable TreeView.
  template<class Enable = std::enable_if_t<!Mutable>>
  constexpr TreeView(const TreeView<!Mutable>& tree) noexcept :
    nodes_(tree.nodes()),
    element_first_(tree.elementFirst()),
    element_last_(tree.elementLast())
  {}

  constexpr std::size_t elementFirst() const noexcept {
    return element_first_;
  }

  constexpr std::size_t elementLast() const noexcept {
    return element_last_;
  }

  constexpr bool empty() const noexcept {
    return nodes_.empty();
  }

  constexpr std::span<value_type> nodes() const noexcept {
    return nodes_;
  }

  constexpr std::size_t pivot() const noexcept {
    return element_first_ + (element_last_ - element_first_) / 2;
  }

  constexpr reference root() const {
    return nodes_.front();
  }

  constexpr TreeView leftChild() const noexcept {
    const std::size_t num_nodes_left = nodes_.size() / 2;
    return TreeView(nodes_.subspan(1, num_nodes_left), element_first_, pivot());
  }

  constexpr TreeView rightChild() const noexcept {
    const std::size_t num_nodes_left = nodes_.size() / 2;
    const std::size_t num_nodes_right = (nodes_.size() - 1) / 2;
    return TreeView(nodes_.subspan(1 + num_nodes_left, num_nodes_right), pivot() + 1, element_last_);
  }

private:
  std::span<value_type> nodes_;
  std::size_t element_first_;
  std::size_t element_last_;
};

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

  // Returns the number of elements represented by the leftmost subtree with root at the specified level.
  // \param num_elements - the total number of elements represented by the tree.
  // \param level - depth of the tree for which to count the number of elements. If level == 0, the
  //                function returns `num_elements`.
  // Time complexity: O(1).
  constexpr std::size_t countElementsInLeftmostSubtree(std::size_t num_elements, std::size_t level) noexcept {
    // f(0) = N
    // f(1) = ceil(f(0)/2) = (N+1)/2
    // f(2) = ceil(f(1)/2) = ((N+1)/2 + 1)/2 = (N+3)/4 = ceil(N/4)
    // f(3) = ceil(f(2)/2) = ((N+3)/4 + 1)/2 = (N+7)/8 = ceil(N/8)
    // f(k) = ceil(N/2^k) = (N + 2^k - 1) / 2^k
    return (num_elements + (static_cast<std::size_t>(1) << level) - 1) >> level;
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
    const std::size_t ratio = (capacity + num_elements - 2) / (num_elements - 1);  // ceil(Nmax/(N-1))
    return ceilLog2(ratio) - 1;
  }

  // Find the depth of the leftmost subtree that represents exactly the specified number of elements.
  // \param capacity_to_find - capacity of the subtree to find.
  // \param capacity - capacity of the full tree.
  // \return depth of the leftmost subtree that represents exactly `capacity_to_find` elements,
  //         or static_cast<std::size_t>(-1) if there is no such subtree.
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

}  // namespace Detail_NS

template<class T>
template<bool Mutable>
class CumulativeHistogram<T>::TreeViewAdvanced {
public:
  using value_type = std::conditional_t<Mutable, T, const T>;
  using reference = value_type&;

  // Expects that 0 < num_elements <= capacity.
  constexpr TreeViewAdvanced(std::span<value_type> nodes,
                             std::size_t num_elements,
                             std::size_t capacity) noexcept :
    TreeViewAdvanced(nodes, num_elements, 0, capacity - 1)
  {}

  // Converting constructor for an immutable TreeView from a mutable TreeView.
  template<class Enable = std::enable_if_t<!Mutable>>
  constexpr TreeViewAdvanced(const TreeViewAdvanced<!Mutable>& tree) noexcept :
    TreeViewAdvanced(tree.nodes(), tree.capacity(), tree.elementFirst(), tree.elementTheoreticalLast())
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

  constexpr TreeViewAdvanced leftChild() const noexcept {
    // The left subtree (if it exists) should always be at full capacity.
    const std::size_t capacity_left = (capacity() + 1) / 2;   // ceil(capacity() / 2)
    const std::size_t num_nodes_left = nodes_.size() / 2;     // ceil((nodes_.size() - 1) / 2)
    return TreeViewAdvanced(nodes_.subspan(1, num_nodes_left),
                            capacity_left, element_first_, pivot());
  }

  constexpr TreeViewAdvanced rightChild() const noexcept {
    const std::size_t num_nodes_left = nodes_.size() / 2;         // ceil((nodes_.size() - 1) / 2)
    const std::size_t capacity_total = capacity();
    const std::size_t capacity_left = (capacity_total + 1) / 2;   // ceil(capacity_total / 2)
    const std::size_t capacity_right = capacity_total / 2;        // floor(capacity_total / 2)
    const std::size_t element_pivot = pivot();
    const std::size_t num_elements_right = num_elements_ - capacity_left;
    // Find the deepest leftmost subtree of the immediate right subtree that represents all
    // elements [element_pivot + 1; element_pivot + num_elements_right].
    const std::size_t level = Detail_NS::findDeepestNodeForElements(num_elements_right, capacity_right);
    const std::size_t capacity_at_level = Detail_NS::countElementsInLeftmostSubtree(capacity_right, level);
    const std::size_t num_nodes_at_level = Detail_NS::countNodesInTree(capacity_at_level);
    // Skip the 0th node because it's the root.
    // Skip the next `num_nodes_left` because they belong to the left subtree.
    // Skip the next `level` nodes because those are nodes between the root of our
    //      "effective" right subtree and the root of the current tree.
    const std::span<value_type> nodes_at_level = nodes_.subspan(1 + num_nodes_left + level, num_nodes_at_level);
    return TreeViewAdvanced(nodes_at_level,
                            num_elements_right, element_pivot + 1, element_pivot + capacity_at_level);
  }

private:
  constexpr TreeViewAdvanced(std::span<value_type> nodes,
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

template<class T>
void CumulativeHistogram<T>::buildTree(std::span<const T> elements, std::span<T> nodes, std::size_t capacity) {
  if (elements.empty()) {
    return;
  }
  const std::size_t level = Detail_NS::findDeepestNodeForElements(elements.size(), capacity);
  const std::size_t capacity_at_level = Detail_NS::countElementsInLeftmostSubtree(capacity, level);
  const std::size_t root_idx = level;
  const std::size_t num_nodes_at_level = Detail_NS::countNodesInTree(capacity_at_level);
  const std::span<T> nodes_at_level = nodes.subspan(root_idx, num_nodes_at_level);
  TreeView<true> tree(nodes_at_level, 0, capacity_at_level - 1);
  //nodes[root_idx - 1] =
  buildTreeImpl(elements, tree);
  // TODO: zero-initialize reserved elements?
}

template<class T>
constexpr T CumulativeHistogram<T>::sumElementsOfEmptyTree(std::span<const T> elements,
                                                           const TreeView<false>& tree) {
  const std::size_t first = tree.elementFirst();
  const std::size_t last = tree.elementLast();
  if (first >= elements.size()) {
    return {};
  }
  if (first == last || last >= elements.size()) {
    return elements[first];
  }
  return elements[first] + elements[last];
}

template<class T>
constexpr T CumulativeHistogram<T>::sumElementsOfFullTree(std::span<const T> elements,
                                                          TreeView<false> tree) {
  T result {};
  while (!tree.empty()) {
    result += tree.root();
    tree = tree.rightChild();
  }
  const std::size_t element_first = tree.elementFirst();
  const std::size_t element_last = tree.elementLast();
  result += elements[element_first];
  if (element_first != element_last) {
    result += elements[element_last];
  }
  return result;
}

template<class T>
T CumulativeHistogram<T>::buildTreeImpl(std::span<const T> elements, const TreeView<true>& tree) {
  // Iterative version
  /*
  if (tree.empty())
  {
    return sumElementsOfEmptyTree(elements, tree);
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
        vars.tree.root() += sumElementsOfEmptyTree(elements, left_child);
      } else {
        // Schedule a call to build the left subtree.
        stack.push_back(StackVariables{ .tree = left_child, .sum_dest = &vars.tree.root(), .done = false });
      }
      // The sum of elements from the right subtree should be added to OUR return value.
      const TreeView<true> right_child = vars.tree.rightChild();
      if (right_child.empty()) {
        *(vars.sum_dest) += sumElementsOfEmptyTree(elements, right_child);
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
  return total_sum + sumElementsOfEmptyTree(elements, t);
  */

  // Two recursive calls
  // Lol, surprisingly, this is the most efficient version.
  if (tree.empty()) {
    return sumElementsOfEmptyTree(elements, tree);
  }
  const T total_sum_left = buildTreeImpl(elements, tree.leftChild());
  tree.root() = total_sum_left;
  return total_sum_left + buildTreeImpl(elements, tree.rightChild());
}

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
CumulativeHistogram<T>::CumulativeHistogram(size_type num_elements):
  elements_(num_elements)
{
  if (num_elements != 0) {
    capacity_ = elements_.capacity();
    // TODO: only construct nodes that are needed to represent the current level.
    nodes_.resize(Detail_NS::countNodesInTree(capacity()));
  }
}

template<class T>
CumulativeHistogram<T>::CumulativeHistogram(size_type num_elements, const T& value):
  CumulativeHistogram(std::vector<T>(num_elements, value))
{
}

template<class T>
CumulativeHistogram<T>::CumulativeHistogram(std::vector<T>&& elements):
  elements_(std::move(elements))
{
  if (elements_.empty()) {
    return;
  }
  // If the input vector already has reserved memory, let's use it.
  capacity_ = elements_.capacity();
  // TODO: only construct nodes that are needed to represent the current level.
  nodes_.resize(Detail_NS::countNodesInTree(capacity()));
  buildTree(elements_, nodes_, capacity());
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
  // Get the capacity of the currently effective tree.
  const size_type capacity_current = capacityCurrent();
  // Reserve new data for elements.
  elements_.reserve(num_elements);
  // Compute the minimum number of nodes in the tree that can represent `num_elements` elements.
  const std::size_t num_nodes_new = Detail_NS::countNodesInTree(num_elements);

  // Check the special case when the tree for num_elements has our current tree as a subtree.
  // In that case there's no need to rebuild the tree - we can just copy our current one.
  const std::size_t level_for_the_original =
    Detail_NS::findLeftmostSubtreeWithExactCapacity(capacity_current, num_elements);
  if (level_for_the_original == static_cast<std::size_t>(-1)) {
    // Not a subtree - just zero-initialize the remaining data and rebuild the tree.
    nodes_.clear();
    // TODO: only construct nodes that are needed to represent the current level.
    nodes_.resize(num_nodes_new);
    buildTree(elements_, nodes_, num_elements);
  } else {
    std::vector<T> new_nodes;
    new_nodes.reserve(num_nodes_new);
    // 1) Zero-initialize reserved "root" nodes.
    new_nodes.insert(new_nodes.end(), level_for_the_original, T{});
    // 2) Just copy the current tree as a subtree of the new one.
    const size_type root_idx_old = getRootIndex();
    const size_type num_nodes_old = Detail_NS::countNodesInTree(capacity_current);
    new_nodes.insert(new_nodes.end(), std::make_move_iterator(nodes_.begin() + root_idx_old),
                                      std::make_move_iterator(nodes_.begin() + root_idx_old + num_nodes_old));
    // 3) Zero-initialize the remaining data.
    // TODO: don't.
    new_nodes.resize(num_nodes_new);
    // 4) Replace old data with new data.
    nodes_ = std::move(new_nodes);
  }
  capacity_ = num_elements;
}

template<class T>
void CumulativeHistogram<T>::clear() noexcept {
  elements_.clear();
  // TODO: call nodes_.clear() instead.
  std::fill(nodes_.begin(), nodes_.end(), T{});
}

template<class T>
void CumulativeHistogram<T>::setZero() {
  std::fill(elements_.begin(), elements_.end(), T{});
  std::fill(nodes_.begin(), nodes_.end(), T{});
}

template<class T>
void CumulativeHistogram<T>::push_back() {
  // Double the capacity if needed.
  if (size() == capacity()) {
    const size_type capacity_new = (capacity() == 0) ? 2 : (capacity() * 2);
    reserve(capacity_new);
  }
  // TODO: this won't work if pop_back() does't clean up after itself. In that
  // case we'll need to initialize the new nodes.

  // Note that the code above ensures that capacity is not 0, so
  // capacityCurrent() != 0. Therefore, if num_elements_ == 0, then the root remains
  // the same.
  const bool should_update_root = (size() == capacityCurrent());
  T total_sum {};
  if (should_update_root) {
    total_sum = totalSum();
  }
  elements_.emplace_back();
  // If the root has changed and the histogram wasn't empty, then we need to
  // initialize the new root node.
  if (should_update_root) {
    // The root stores the sum of elements from the left subtree.
    const size_type root_idx = getRootIndex();
    // We have added a 0-initialized element, so the total sum hasn't changed.
    nodes_[root_idx] = total_sum;
  }
}

template<class T>
void CumulativeHistogram<T>::push_back(const T& value) {
  // Double the capacity if needed.
  if (size() == capacity()) {
    const size_type capacity_new = (capacity() == 0) ? 2 : (capacity() * 2);
    reserve(capacity_new);
  }
  // If the tree is full at its current capacity, we need to add a new root and
  // initialize it with the total sum.
  if (size() == capacityCurrent()) {
    // Current capacity cannot be 0 - we've just ensured that the capacity is not 0,
    // hence the current capacity is not 0 either.
    T total_sum = totalSum();
    elements_.push_back(value);
    const size_type root_idx_new = getRootIndex();
    nodes_[root_idx_new] = std::move(total_sum);
    // The new element is added to the new right subtree, which doesn't have any nodes yet
    // because it only represents 1 element.
    return;
  }

  std::span<const T> tmp_elements{ elements_ };
  std::size_t tmp_capacity = capacityCurrent();
  std::size_t tmp_num_nodes = numNodesCurrent();
  std::size_t tmp_root_idx = getRootIndex();
  while (tmp_num_nodes != 0) {
    // If the current tree has at least 1 nodes, then its left subtree must be full.
    // Since we already know that the main tree is not full, it is safe to assume that the immediate
    // right subtree is neither empty nor full.
    const std::size_t num_nodes_left = tmp_num_nodes / 2;      // ceil((tmp_num_nodes - 1) / 2)
    const std::size_t capacity_left = (tmp_capacity + 1) / 2;  // ceil(tmp_capacity / 2)
    const std::size_t capacity_right = tmp_capacity / 2;       // floor(tmp_capacity / 2)
    const std::span<const T> elements_right = tmp_elements.subspan(capacity_left);
    // Find the deepest leftmost subtree of the immediate right subtree that represents all
    // of `elements_right`.
    const std::size_t level = Detail_NS::findDeepestNodeForElements(elements_right.size(), capacity_right);
    // If the effective right subtree is the immediate right subtree, just switch to the immediate right subtree.
    if (level == 0) {
      tmp_elements = elements_right;
      tmp_capacity = capacity_right;
      tmp_num_nodes = (tmp_num_nodes - 1) / 2;  // floor((tmp_num_nodes - 1) / 2)
      tmp_root_idx += (1 + num_nodes_left);
      continue;
    }
    // Otherwise, check if the effective right subtree is full.
    const std::size_t capacity_at_level = Detail_NS::countElementsInLeftmostSubtree(capacity_right, level);
    const std::size_t num_nodes_at_level = Detail_NS::countNodesInTree(capacity_at_level);
    if (elements_right.size() == capacity_at_level) {
      // Add a new root for the effective right subtree.
      const std::size_t root_idx_old = tmp_root_idx + 1 + num_nodes_left + level;
      const std::size_t root_idx_new = tmp_root_idx + 1 + num_nodes_left + level - 1;
      // Compute the sum of all elements in the effective right subtree.
      const std::span<T> nodes_at_level = std::span<T>{ nodes_ }.subspan(root_idx_old, num_nodes_at_level);
      TreeView<true> tree_at_level(nodes_at_level, 0, capacity_at_level - 1);
      const T sum_at_level = sumElementsOfFullTree(elements_right, tree_at_level);
      nodes_[root_idx_new] = sum_at_level;
      // The new element is added to the right subtree of the newly constructed tree. This subtree doesn't
      // have any nodes yet, because it only represents 1 element.
      elements_.push_back(value);
      return;
    }
    // OK, the effective right subtree is not the immediate right subtree and it's not full. Let's switch to it.
    tmp_elements = elements_right;
    tmp_capacity = capacity_at_level;
    tmp_num_nodes = num_nodes_at_level;
    tmp_root_idx += (1 + num_nodes_left + level);
    continue;
  }
  // If we are here, then we've reached the deepmost subtree which doesn't have any nodes, so it can only represent
  // either 1 or 2 elements. However, it cannot be at its full capacity, and it cannot be empty, which means that it
  // can only represent 2 elements and currently store 1. So we simply add the new element.
  elements_.push_back(value);
}

template<class T>
void CumulativeHistogram<T>::pop_back() {
  if (empty()) {
    throw std::logic_error("CumulativeHistogram::pop_back(): there are no elements left to remove.");
  }
  // Note that this works even if T is an unsigned type, thanks to modular arithmetic.
  const T diff = static_cast<T>(T{} - elements_[size() - 1]);
  increment(size() - 1, diff);
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
    // TODO: replace with a decent implementation - this one is fucking terrible.
    const size_type elements_to_remove = size() - num_elements;
    for (size_type i = 0; i < elements_to_remove; ++i) {
      pop_back();
    }
    return;
  }
  // Append N'-N elements if N < N'.
  // Allocate new data if needed.
  reserve(num_elements);
  // Save the old index of the root. Note that if we're here, then
  //   0 <= old_num_elements < new_num_elements <= capacity_
  const size_type old_root_idx = getRootIndex();
  // TODO: this won't work if pop_back() does't clean up after itself.
  T total_sum {};
  if (!empty()) {
    total_sum = totalSum();
  }
  elements_.resize(num_elements);
  // Compute the new index of the root.
  const size_type root_idx = getRootIndex();
  // If the root has changed, then we need to initialize the new nodes [root_idx_ - 1; old_root_idx - 1)
  // with the total sum.
  if (root_idx != old_root_idx) {
    const auto iter_root_new = nodes_.begin() + root_idx;
    const auto iter_root_old = nodes_.begin() + old_root_idx;
    // TODO: this might be broken if there were no nodes initially.
    std::fill(iter_root_new, iter_root_old, total_sum);
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
  const std::span<T> nodes = std::span<T>{nodes_}.subspan(root_idx, numNodesCurrent());
  TreeView<true> tree(nodes, 0, capacityCurrent() - 1);
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
  //if (k == size() - 1) {
  //  return totalSum();
  //}
  T result {};
  // Tree representing the elements [0; N).
  const size_type root_idx = getRootIndex();
  const std::span<const T> nodes = std::span<const T>{nodes_}.subspan(root_idx, numNodesCurrent());
  TreeViewAdvanced<false> tree(nodes, size(), capacityCurrent());
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
  // Add values of existing elements from the last tree.
  const std::size_t element_first = tree.elementFirst();
  const std::size_t element_last = tree.elementTheoreticalLast();
  if (element_first <= k) {
    result += elements_[element_first];
    if (element_first != element_last && element_last <= k) {
      result += elements_[element_last];
    }
  }
  return result;  
}

template<class T>
constexpr T CumulativeHistogram<T>::totalSum() const {
  if (empty()) {
    throw std::logic_error("CumulativeHistogram::totalSum(): the histogram is empty.");
  }
  return prefixSum(size() - 1);
}

template<class T>
template<class Compare>
std::pair<typename CumulativeHistogram<T>::const_iterator, T>
CumulativeHistogram<T>::lowerBoundImpl(const T& value, Compare cmp) const {
  // Terminate if there are no elements.
  if (empty()) {
    return { end(), T{} };
  }
  // Terminate if cmp(totalSum(), value) is true - in this case there is no such
  // index k that !cmp(prefixSum(k), value).
  // TODO: don't compute the total sum.
  const T total_sum = totalSum();
  if (cmp(total_sum, value)) {
    return { end(), T{} };
  }
  T prefix_sum_before_lower {};
  T prefix_sum_upper = total_sum;
  // Tree representing the elements [k_lower; k_upper] = [0; N-1].
  const size_type root_idx = getRootIndex();
  const std::span<const T> nodes = std::span<const T>{ nodes_ }.subspan(root_idx, numNodesCurrent());
  TreeView<false> tree(nodes, 0, capacityCurrent() - 1);
  while (!tree.empty()) {
    // TODO: be careful when checking tree.root() - it's possible that middle >= N,
    // and if you decide not to initialize nodes for non-existent nodes, then accessing tree.root()
    // can be UB. In that case you should just go to the left child unconditionally.
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
  const std::size_t k_lower = tree.elementFirst(); // cmp(prefixSum(i), value) == true for all i < k_lower
  const std::size_t k_upper = tree.elementLast();  // cmp(prefixSum(k_upper), value) == false
  prefix_sum_before_lower += elements_[k_lower];
  if (!cmp(prefix_sum_before_lower, value)) {
    return { begin() + k_lower, prefix_sum_before_lower };
  }
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
