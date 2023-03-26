#pragma once

#include <bit>
#include <numeric>
#include <stdexcept>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace CumulativeHistogram_NS {

// This class allows to efficiently compute prefix sums for a dynamic array of elements
// by offering a compromise between 2 naive solutions:
// +-----------------------------------+-------------------+------------+-------+
// | Solution                          | Modify an element | Prefix sum | Space |
// +-----------------------------------+-------------------+------------+-------+
// | 1. Store the elements in an array | O(1)              | O(N)       | O(N)  |
// +-----------------------------------+-------------------+------------+-------+
// | 2. Store the prefix sums in array | O(N)              | O(1)       | O(N)  |
// +-----------------------------------+-------------------+------------+-------+
// | 3. CumulativeHistogram            | O(logN)           | O(logN)    | O(N)  |
// +-----------------------------------+-------------------+------------+-------+
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
  constexpr CumulativeHistogram() noexcept = default;

  // Constructs a cumulative histogram for N zero-initialized elements.
  // Time complexity: O(N).
  explicit CumulativeHistogram(size_type num_elements);

  // Constructs a cumulative histogram for the specified elements.
  // TODO: remove this contructor if you decide not to store data in std::vector.
  explicit CumulativeHistogram(std::vector<T>&& elements);

  // Constructs a cumulative histogram for the specified elements.
  // Time complexity: O(N), where N is the distance between first and last.
  template<class Iter>
  CumulativeHistogram(Iter first, Iter last);

  // Returns an iterator to the first element.
  constexpr const_iterator begin() const noexcept;

  // Returns an iterator past the last element.
  constexpr const_iterator end() const noexcept;

  // Returns a reverse iterator to the last element.
  constexpr const_reverse_iterator rbegin() const noexcept;

  // Returns a reverse iterator preceding the first element.
  constexpr const_reverse_iterator rend() const noexcept;

  // Returns true if the number of elements is greater than 0, false otherwise.
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
  // TODO: could be O(1) if we require that T is an arithmetic type. However, that woukd be too restrictive:
  //       e.g., there's nothing wrong with CumulativeHistogram<std::complex>.
  void clear() noexcept;

  // Add a zero-initialized element to the end.
  // Time complexity: amortized O(1).
  void push_back();

  // Add an element to the end.
  // Time complexity: amortized O(logN).
  // Note: it's O(logN) not because we need to update O(logN) nodes (AFAIU, it's possible to
  // only update O(1) nodes), but because we need to traverse the tree).
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
  // Time complexity: O(1).
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
  // Implementation details.
  class Detail;

  // Returns the maximum number of elements that can represented by the current tree.
  //   num_elements <= capacityCurrent() <= capacity_
  constexpr size_type capacityCurrent() const noexcept;

  // Returns the number of nodes in the current tree (not including the element storing the total sum).
  constexpr size_type numNodesCurrent() const noexcept;

  // Shared implementation for lowerBound() and upperBound().
  // If Upper == false, finds the first element k for which prefixSum(k) >= value,
  //          otherwise finds the first element k for which prefixSum(k) > value.
  // In both cases objects of type T are compared only using operator<.
  template<bool Upper>
  std::pair<const_iterator, T> lowerBoundImpl(const T& value) const;

  template<bool Mutable>
  class TreeView;

  // Internal data.
  // TODO: consider not using std::vector.
  // Rationale: if the histogram is at full capacity and we want to add more element(s),
  // we need to allocate more memory. However, we won't just copy current data via a
  // single memcpy() call: in the new memory there will be a "gap" between data for the
  // elements and data for the tree.
  // We can still use std::vector for storage, but we don't need its insert/erase API.
  // In fact, unique_ptr<T[]> will do fine.
  std::vector<T> data_;
  // The number of elements N.
  size_type num_elements_ = 0;
  // Current capacity Nmax.
  size_type capacity_ = 0;
  // Index of the root node.
  // * This is always equal to
  //   capacity_ + 1 + Detail::findDeepestNodeForElements(num_elements_, capacity_)
  // * data_[root_idx_] may be out of range if the tree has 0 nodes.
  // * data_[root_idx_-1] stores the total sum of elements [0; num_elements_) unless
  //   the histogram is empty.
  // TODO: consider removing and calling Detail::findDeepestNodeForElements() when needed.
  size_type root_idx_ = 1;
};

// ----==== Implementation ====----

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
// +------+------+------+------+------+------------+-----------+--------+--------+--------+------+--------+
// |   0  |   1  |   2  | .... |  N-1 | .......... |    Nmax   | Nmax+1 | Nmax+2 | Nmax+3 | ...  | Nmax+M |
// +------+------+------+------+------+------------+-----------+--------+--------+--------+------+--------+
// |  x0  |  x1  |  x2  | .... | xN-1 | ...free... | total_sum |   n0   |   n1   |   n2   | .... |  nM-1  |
// +------+------+------+------+------+------------+-----------+--------+--------+--------+------+--------+
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

template<class T>
class CumulativeHistogram<T>::Detail {
 public:
  // Returns floor(log2(x)).
  static std::size_t floorLog2(std::size_t value) noexcept {
    return std::bit_width(value) - 1;
  }

  // Returns ceil(log2(x)).
  static std::size_t ceilLog2(std::size_t value) noexcept {
    const bool is_power_of_2 = std::has_single_bit(value);
    const std::size_t floor_log2 = floorLog2(value);
    return is_power_of_2 ? floor_log2 : (floor_log2 + 1);
  }

  // Returns the total number of nodes in the auxiliary tree for
  // CumulativeHistogram with the specified number of elements.
  static constexpr std::size_t countNodesInTree(std::size_t num_elements) noexcept {
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
  static constexpr std::size_t countElementsInLeftmostSubtree(std::size_t num_elements, std::size_t level) noexcept {
    // f(0) = N
    // f(1) = ceil(f(0)/2) = (N+1)/2
    // f(2) = ceil(f(1)/2) = ((N+1)/2 + 1)/2 = (N+3)/4 = ceil(N/4)
    // f(3) = ceil(f(2)/2) = ((N+3)/4 + 1)/2 = (N+7)/8 = ceil(N/8)
    // f(k) = ceil(N/2^k) = (N + 2^k - 1) / 2^k
    return (num_elements + (static_cast<std::size_t>(1) << level) - 1) >> level;
  }

  // Returns the depth of the deepest node containing the elements [0; num_elements)
  // in the optimal tree representing the elements [0; capacity).
  // Expects that num_elements <= capacity.
  // The depth of the root is 0.
  // Time complexity: O(1).
  static constexpr std::size_t findDeepestNodeForElements(std::size_t num_elements,
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
    //    Nmax/(N-1) > 2^k
    // <=>
    //    k < log2(Nmax/(N-1)
    // <=>
    //    The greatest such k is ceil(log2(Nmax/(N-1)))-1
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

    const std::size_t ratio = (capacity + num_elements - 2) / (num_elements - 1);  // ceil(Nmax/(N-1))
    return ceilLog2(ratio) - 1;
  }

  // Build the tree for the given elements.
  // Expects that:
  // 1) 0 <= num_elements <= capacity
  // 2) data.size() == capacity + 1 + countNodesInTree(capacity)
  // Time complexity: O(num_elements).
  // TODO: consider changing the signature to
  //     void buildTree(std::span<const T> elements, std::span<T> nodes, std::size_t capacity)
  static void buildTree(std::span<T> data, std::size_t num_elements, std::size_t capacity) {
    if (num_elements == 0) {
      return;
    }
    const std::span<const T> elements = data.first(num_elements);
    const std::size_t level = findDeepestNodeForElements(num_elements, capacity);
    const std::size_t capacity_at_level = countElementsInLeftmostSubtree(capacity, level);
    const std::size_t root_idx = capacity + 1 + level;
    const std::size_t num_nodes_at_level = countNodesInTree(capacity_at_level);
    const std::span<T> nodes = data.subspan(root_idx, num_nodes_at_level);
    TreeView<true> tree(nodes, 0, capacity_at_level - 1);
    data[root_idx - 1] = buildTreeImpl(elements, tree);
    // TODO: zero-initialize reserved elements?
  }

 private:
  // This function is intended to be called for empty trees (trees without nodes).
  // Such trees always represent either 1 or 2 elements, so there's no need for a loop.
  static constexpr T sumElementsOfEmptyTree(std::span<const T> elements, const TreeView<false>& tree) {
    const std::size_t first = tree.elementFirst();
    const std::size_t last = tree.elementLast();
    if (first == last) {
      return elements[first];
    }
    return elements[first] + elements[last];
  }

  static T buildTreeImpl(std::span<const T> elements, const TreeView<true>& tree) {
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
    // Reserve memory all nodes.
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
};

template<class T>
constexpr
typename CumulativeHistogram<T>::size_type
CumulativeHistogram<T>::capacityCurrent() const noexcept {
  // Determine the depth of the current root node in the "full" tree.
  // Note that there aren't actually any nodes if capacity_ < 3. In that case root_idx_
  // should still equal capacity_ + 1 (even though data[root_idx_] is out of range).
  // level will be 0, and this function will simply return capacity_.
  const size_type level = root_idx_ - capacity_ - 1;
  return Detail::countElementsInLeftmostSubtree(capacity_, level);
}

template<class T>
constexpr std::size_t CumulativeHistogram<T>::numNodesCurrent() const noexcept {
  return Detail::countNodesInTree(capacityCurrent());
}

template<class T>
CumulativeHistogram<T>::CumulativeHistogram(size_type num_elements):
  data_(num_elements ? num_elements + 1 + Detail::countNodesInTree(num_elements) : 0),
  num_elements_(num_elements),
  capacity_(num_elements),
  root_idx_(capacity_ + 1)
{
  // This function constructs a CumulativeHistogram that is at its full capacity.
  // data_[num_elements] stores the total sum.
  // data_[num_elements+1] stores the root of the tree.
}

template<class T>
CumulativeHistogram<T>::CumulativeHistogram(std::vector<T>&& elements):
  data_(std::move(elements)),
  num_elements_(data_.size()),
  capacity_(num_elements_),
  root_idx_(capacity_ + 1)
{
  // TODO: reuse capacity of the input vector.
  // Let N = elements.size(), C = elements.capacity().
  //   * If C < N + 1 + countNodesInTree(N), then we need to allocate memory anyway,
  //     so let's not allocate more than necessary.
  //   * If C = N + 1 + countNodesInTree(N), then there's no need to allocate more memory,
  //     but CumulativeHistogram is at its full capacity.
  //   * If C = N + 1 + countNodesInTree(N), then our capacity_ might be greater than N.
  //     However, this requires solving the optimization problem:
  //         find the largest Nmax, such that Nmax + 1 + countNodesInTree(Nmax) <= C.
  //     It can be solved with binary search (the solution is somewhere between [N; C),
  //     but I'd prefer an algorithm with O(1) time complexity.
  data_.resize(num_elements_ + 1 + Detail::countNodesInTree(num_elements_));
  Detail::buildTree(data_, num_elements_, capacity_);
}

template<class T>
template<class Iter>
CumulativeHistogram<T>::CumulativeHistogram(Iter first, Iter last)
{
  using iterator_categoty = typename std::iterator_traits<Iter>::iterator_category;
  size_type new_data_size;
  if constexpr (std::is_same_v<iterator_categoty, std::input_iterator_tag>) {
    // If Iter does not allow multipass, we need to materialize it first.
    data_.assign(first, last);
    num_elements_ = data_.size();
    new_data_size = num_elements_ + 1 + Detail::countNodesInTree(num_elements_);
  } else {
    num_elements_ = std::distance(first, last);
    new_data_size = num_elements_ + 1 + Detail::countNodesInTree(num_elements_);
    data_.reserve(new_data_size);
    data_.insert(data_.end(), first, last);
  }
  // Append new default-constructed elements and our auxiliary counters.
  data_.resize(new_data_size);
  capacity_ = num_elements_;
  root_idx_ = capacity_ + 1;
  Detail::buildTree(data_, num_elements_, capacity_);
}

template<class T>
constexpr
typename CumulativeHistogram<T>::const_iterator CumulativeHistogram<T>::begin() const noexcept {
  return data_.data();
}

template<class T>
constexpr
typename CumulativeHistogram<T>::const_iterator CumulativeHistogram<T>::end() const noexcept {
  return data_.data() + num_elements_;
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
  return num_elements_ == 0;
}

template<class T>
constexpr
typename CumulativeHistogram<T>::size_type
CumulativeHistogram<T>::size() const noexcept {
  return num_elements_;
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
  // Construct new data.
  const size_type new_data_size = num_elements + 1 + Detail::countNodesInTree(num_elements);
  std::vector<T> new_data;
  new_data.reserve(new_data_size);
  new_data.insert(new_data.end(), std::make_move_iterator(data_.begin()),
                                  std::make_move_iterator(data_.begin() + num_elements_));
  new_data.resize(new_data_size);
  // TODO: Check the special case when the tree for num_elements has our current tree as a subtree.
  // In that case there's no need to rebuild the tree - we can just copy our current one.
  Detail::buildTree(new_data, num_elements_, num_elements);
  // Replace old data with new data.
  data_ = std::move(new_data);
  capacity_ = num_elements;
  root_idx_ = capacity_ + 1 + Detail::findDeepestNodeForElements(num_elements_, capacity_);
}

template<class T>
void CumulativeHistogram<T>::clear() noexcept {
  data_.clear();
  num_elements_ = 0;
  root_idx_ = 1;
}

template<class T>
void CumulativeHistogram<T>::setZero() {
  std::fill(data_, T{});
}

template<class T>
void CumulativeHistogram<T>::push_back() {
  const bool was_empty = empty();
  // Double the capacity if needed.
  if (num_elements_ + 1 > capacity_) {
    const size_type capacity_new = capacity_ == 0 ? 1 : (capacity_ * 2);
    reserve(capacity_new);
  }
  // TODO: this won't work if pop_back() does't clean up after itself. In that
  // case we'll need to initialize the new nodes.
  ++num_elements_;
  if (num_elements_ > capacityCurrent()) {
    // Update the root.
    // This is almost always equivalent to --root_idx_, but not if the histogram had 0
    // nodes before push_back(). Instead of tracking the edge cases, we can just recompute
    // the index of the root.
    root_idx_ = capacity_ + 1 + Detail::findDeepestNodeForElements(num_elements_, capacity_);
    // Initialize the total sum with the sum of elements from the left subtree.
    if (!was_empty) {
      data_[root_idx_ - 1] = data_[root_idx_];
    }
  }
}

template<class T>
void CumulativeHistogram<T>::pop_back() {
  if (empty()) {
    throw std::logic_error("CumulativeHistogram::pop_back(): there are no elements left to remove.");
  }
  // Note that this works even if T is an unsigned type, thanks to modular arithmetic.
  const T diff = static_cast<T>(T{} - data_[num_elements_ - 1]);
  increment(num_elements_ - 1, diff);
  // The number of elements that the current tree represents.
  const size_type capacity_current = capacityCurrent();
  // The number of nodes in the current tree.
  const size_type nodes_current = Detail::countNodesInTree(capacity_current);
  --num_elements_;
  if (nodes_current > 0) {
    const size_type num_elements_in_left_subtree = (capacity_current + 1) / 2;
    // If all "real" elements are now in the left subtree, declare the left subtree as the new tree.
    if (num_elements_ <= num_elements_in_left_subtree) {
      ++root_idx_;
    }
  }
}

template<class T>
void CumulativeHistogram<T>::resize(size_type num_elements) {
  // Do nothing if N == N'
  if (num_elements_ == num_elements) {
    return;
  }
  // Remove the last N-N' elements if N > N'.
  if (num_elements_ > num_elements) {
    // TODO: replace with a decent implementation - this one is fucking terrible.
    const size_type elements_to_remove = num_elements_ - num_elements;
    for (size_type i = 0; i < elements_to_remove; ++i) {
      pop_back();
    }
    return;
  }
  // Append N'-N elements if N < N'.
  if (capacity() >= num_elements) {
    // TODO: this won't work if pop_back() does't clean up after itself.
    num_elements_ = num_elements;
    // TODO: I suppose the new value for root_idx_ could be computed in O(1), but
    // we need to initialize the new nodes anyway, so the loop cannot be avoided.
    while (num_elements_ > capacityCurrent()) {
      // Update the root.
      // TODO: don't update root_idx_ if there were 0 nodes before this iteration.
      --root_idx_;
      // Initialize the total sum with the sum of elements from the left subtree.
      data_[root_idx_ - 1] = data_[root_idx_];
    }
  } else {
    // TODO: check if the tree for num_elements has our current tree as a subtree. If it does,
    // then we can just std::copy() the current tree instead of building a new one.

    // Allocate new data.
    std::vector<T> new_data;
    const size_type new_data_size = num_elements + 1 + Detail::countNodesInTree(num_elements);
    new_data.reserve(new_data_size);
    // Copy current elements.
    new_data.insert(new_data.end(), data_.begin(), data_.begin() + num_elements_);
    // Append new default-constructed elements and our auxiliary counters.
    new_data.resize(new_data_size);
    // TODO: don't replace the old data_ with new_data until the new tree is built -
    // resize() should have a strong exception guarantee.
    Detail::buildTree(new_data, num_elements, num_elements);
    // Replace old data with new data.
    data_ = std::move(new_data);
    num_elements_ = num_elements;
    capacity_ = num_elements;
    root_idx_ = capacity_ + 1;
  }
}

template<class T>
constexpr std::span<const T> CumulativeHistogram<T>::elements() const noexcept {
  return std::span<const T>{data_.data(), num_elements_};
}

template<class T>
typename CumulativeHistogram<T>::const_reference
CumulativeHistogram<T>::element(size_type k) const {
  if (k >= num_elements_) {
    throw std::out_of_range("CumulativeHistogram::element(): k is out of range.");
  }
  return data_[k];
}

template<class T>
void CumulativeHistogram<T>::increment(size_type k, const T& value) {
  if (k >= num_elements_) {
    throw std::out_of_range("CumulativeHistogram::increment(): k is out of range.");
  }
  // Tree representing the elements [0; N).
  const std::span<T> nodes = std::span<T>{data_}.subspan(root_idx_, numNodesCurrent());
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
  // Update the total sum.
  data_[root_idx_ - 1] += value;
  // Update the element itself.
  data_[k] += value;
}

template<class T>
T CumulativeHistogram<T>::prefixSum(size_type k) const {
  if (k >= num_elements_) {
    throw std::out_of_range("CumulativeHistogram::prefixSum(): k is out of range.");
  }
  // Special case for the total sum.
  if (k == num_elements_ - 1) {
    return data_[root_idx_ - 1];
  }
  T result {};
  // Tree representing the elements [0; N).
  const std::span<const T> nodes = std::span<const T>{data_}.subspan(root_idx_, numNodesCurrent());
  TreeView<false> tree(nodes, 0, capacityCurrent() - 1);
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
  result += data_[k];
  return result;  
}

template<class T>
constexpr T CumulativeHistogram<T>::totalSum() const {
  if (empty()) {
    throw std::logic_error("CumulativeHistogram::totalSum(): the histogram is empty.");
  }
  return data_[root_idx_ - 1];
}

template<class T>
template<bool Upper>
std::pair<typename CumulativeHistogram<T>::const_iterator, T>
CumulativeHistogram<T>::lowerBoundImpl(const T& value) const {
  struct LessEqual {
    bool operator()(const T& lhs, const T& rhs) const { return !(rhs < lhs); }
  };
  // Use strict comparison for lowerBound, non-strict comparison for upperBound.
  using Compare = std::conditional_t<Upper, LessEqual, std::less<T>>;
  const Compare is_less;

  // Check if there is an index k for which prefixSum(k) >= value.
  if ((num_elements_ == 0) || is_less(data_[root_idx_ - 1], value)) {
    return { end(), T{} };
  }
  T prefix_sum_before_lower {};
  T prefix_sum_upper = data_[root_idx_ - 1];
  // Tree representing the elements [k_lower; k_upper] = [0; N-1].
  const std::span<const T> nodes = std::span<const T>{ data_ }.subspan(root_idx_, numNodesCurrent());
  TreeView<false> tree(nodes, 0, capacityCurrent() - 1);
  while (!tree.empty()) {
    // TODO: be careful when checking tree.root() - it's possible that middle >= N,
    // and if you decide not to initialize nodes for non-existent nodes, then accessing tree.root()
    // can be UB. In that case you should just go to the left child unconditionally.
    // The root of the tree stores the sum of all elements [k_lower; middle].
    // Sum of elements [0; middle]
    T prefix_sum_middle = prefix_sum_before_lower + tree.root();
    if (is_less(prefix_sum_middle, value)) {
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
  const std::size_t k_lower = tree.elementFirst(); // prefixSum(i) < value for all i < k_lower
  const std::size_t k_upper = tree.elementLast();  // prefixSum(k_upper) > value
  prefix_sum_before_lower += data_[k_lower];
  if (!is_less(prefix_sum_before_lower, value)) {
    return { begin() + k_lower, prefix_sum_before_lower };
  }
  return { begin() + k_upper, prefix_sum_upper };
}

template<class T>
std::pair<typename CumulativeHistogram<T>::const_iterator, T>
CumulativeHistogram<T>::lowerBound(const T& value) const {
  return lowerBoundImpl<false>(value);
}

template<class T>
std::pair<typename CumulativeHistogram<T>::const_iterator, T>
CumulativeHistogram<T>::upperBound(const T& value) const {
  return lowerBoundImpl<true>(value);
}

}  // namespace CumulativeHistogram_NS
