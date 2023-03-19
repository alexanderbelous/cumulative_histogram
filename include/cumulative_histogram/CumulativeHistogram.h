#include <bit>
#include <numeric>
#include <stdexcept>
#include <span>
#include <type_traits>
#include <vector>

namespace CumulativeHistogram_NS {

// This class allows to efficiently compute partial sums for a fixed-size array of mutable elements.
// The naive solution (re-computing the partial sums whenever an element is modified) has O(N) time complexity,
// which can be inefficient if both operations (modifying an element and computing a partial sum) are used often.
//
// Space complexity: O(N), where N is the number of elements.
// The overhead is approximately 50% (i.e. the class stores roughly N/2 extra counters).
template<class T>
class CumulativeHistogram {
 public:
  // Constructs an empty histogram.
  constexpr CumulativeHistogram() noexcept = default;

  // Constructs a cumulative histogram for N elements.
  explicit CumulativeHistogram(std::size_t num_elements);

  // Constructs a cumulative histogram for the specified elements.
  explicit CumulativeHistogram(std::vector<T>&& elements);

  // Constructs a cumulative histogram for the specified elements.
  template<class Iter>
  explicit CumulativeHistogram(Iter first, Iter last);

  // Sets the values of all elements to 0.
  // Time complexity: O(N).
  void setZero();

  // Sets the values of all elements to the specified value.
  // Time complexity: O(N).
  void fill(const T& value);

  // Returns the number of elements in the histogram.
  constexpr std::size_t numElements() const noexcept;

  // Access all elements.
  // Time complexity: O(1).
  constexpr std::span<const T> elements() const noexcept;

  // Access the specified element.
  // Throws std::out_of_range if k >= numElements().
  // Time complexity: O(1).
  const T& element(std::size_t k) const;

  // Increment the specified element.
  // Throws std::out_of_range if k >= numElements().
  // Time complexity: O(log2(N)).
  void increment(std::size_t k, const T& value = 1);

  // Returns the partial sum of the first K elements.
  // Throws std::out_of_range if k >= numElements().
  // Time complexity: O(log2(N)).
  T partialSum(std::size_t k) const;

  // Returns the total sum of all elements, or 0 if the histogram is empty.
  // Time complexity: O(1).
  constexpr T totalSum() const noexcept;

  // Find the first element k, for which partialSum(k) is not less than the specified value.
  // Time complexity: O(log2(N)).
  // TODO: implement.

 private:
  // Returns the total number of nodes in the auxiliary tree for
  // CumulativeHistogram with the specified number of elements.
  static constexpr std::size_t countNodesInTree(std::size_t num_elements) noexcept;

  // Rebuilds the tree data.
  // Time complexity: O(N).
  void rebuildTree() noexcept;

  template<bool Mutable>
  class TreeView;

  T buildTreeImpl(const TreeView<true>& tree) noexcept;

  // Array of N+M counters, where the first N are the actual elements,
  // and the remaining M counters are the nodes of the implicit tree.
  std::vector<T> data_;
  // The number of elements N.
  std::size_t num_elements_;
  // Sum of all elements [0; N).
  T total_sum_ = {};
};

// ----==== Implementation ====----

// CumulativeHistogram stores auxiliary counters for sums of certain elements.
// These counters are updated whenever the respective elements are modified -
// this is why CumulativeHistogram::increment() has O(log2(N)) time complexity.
// However, having these sums precomputed also allows efficient computation of the partial sums.
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
// Any partial sum can be computed by going from the root of the tree to the leaf:
// +------------------------------------------------------------------------------------------+
// |  s0  |  s1  |  s2  |  s3   |  s4   |  s5  |  s6   |  s7   |  s8   |  s9      |  s10      |
// +------------------------------------------------------------------------------------------+
// |  x0  |  n2  |  n1  | n1+x3 | n1+n3 |  n0  | n0+x6 | n0+n5 | n0+n4 | n0+n4+x9 | total_sum |
// +------------------------------------------------------------------------------------------+
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

  constexpr std::size_t elementFirst() const noexcept {
    return element_first_;
  }

  constexpr std::size_t elementLast() const noexcept {
    return element_last_;
  }

  constexpr bool empty() const noexcept {
    return nodes_.empty();
  }

  constexpr std::size_t pivot() const noexcept {
    return element_first_ + (element_last_ - element_first_) / 2;
  }

  reference root() const {
    return nodes_.front();
  }

  constexpr TreeView leftChild() const {
    const std::size_t num_nodes_left = nodes_.size() / 2;
    return TreeView(nodes_.subspan(1, num_nodes_left), element_first_, pivot());
  }

  constexpr TreeView rightChild() const {
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
constexpr std::size_t CumulativeHistogram<T>::countNodesInTree(std::size_t num_elements) noexcept {
  // This number can be compute via a recurrent relation:
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

// TODO: replace recursion with a loop.
template<class T>
T CumulativeHistogram<T>::buildTreeImpl(const TreeView<true>& tree) noexcept {
  // Every node represents at least 3 elements - we don't store smaller nodes.
  // Thus, any non-empty tree has elements in both left and right subtrees, even
  // if these subtrees don't actually have nodes.
  if (tree.empty()) {
    const std::size_t element_first = tree.elementFirst();
    const std::size_t element_last = tree.elementLast();
    const std::size_t num_elements = element_last - element_first + 1;
    std::span<const T> elements{ data_.data() + element_first, num_elements };
    return std::accumulate(elements.begin(), elements.end(), T{});
  }
  const T total_sum_left = buildTreeImpl(tree.leftChild());
  const T total_sum_right = buildTreeImpl(tree.rightChild());
  tree.root() = total_sum_left;
  return total_sum_left + total_sum_right;
}

template<class T>
void CumulativeHistogram<T>::rebuildTree() noexcept {
  const std::span<T> nodes = std::span<T>{data_}.subspan(num_elements_);
  TreeView<true> tree(nodes, 0, num_elements_ - 1);
  total_sum_ = buildTreeImpl(tree);
}

template<class T>
CumulativeHistogram<T>::CumulativeHistogram(std::size_t num_elements):
  data_(num_elements + countNodesInTree(num_elements)),
  num_elements_(num_elements)
{}

template<class T>
CumulativeHistogram<T>::CumulativeHistogram(std::vector<T>&& elements):
  data_(std::move(elements)),
  num_elements_(data_.size())
{
  data_.resize(num_elements_ + countNodesInTree(num_elements_));
  rebuildTree();
}

// TODO: if Iter allows multipass, we should only do 1 memory allocation.
template<class T>
template<class Iter>
CumulativeHistogram<T>::CumulativeHistogram(Iter first, Iter last):
  data_(first, last),
  num_elements_(data_.size())
{
  data_.resize(num_elements_ + countNodesInTree(num_elements_));
  rebuildTree();
}

template<class T>
void CumulativeHistogram<T>::setZero() {
  std::fill(data_, T{});
}

template<class T>
constexpr std::size_t CumulativeHistogram<T>::numElements() const noexcept {
  return num_elements_;
}

template<class T>
constexpr std::span<const T> CumulativeHistogram<T>::elements() const noexcept {
  return std::span<const T>{data_.data(), num_elements_};
}

template<class T>
const T& CumulativeHistogram<T>::element(std::size_t k) const {
  if (k >= num_elements_) {
    throw std::out_of_range("CumulativeHistogram::element(): k is out of range.");
  }
  return data_.data()[k];
}

template<class T>
void CumulativeHistogram<T>::increment(std::size_t k, const T& value) {
  if (k >= num_elements_) {
    throw std::out_of_range("CumulativeHistogram::increment(): k is out of range.");
  }
  // Tree representing the elements [0; N).
  const std::span<T> nodes = std::span<T>{data_}.subspan(num_elements_);
  TreeView<true> tree(nodes, 0, num_elements_ - 1);
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
  data_[k] += value;
  // Update the total sum.
  total_sum_ += value;
}

template<class T>
T CumulativeHistogram<T>::partialSum(std::size_t k) const {
  if (k >= num_elements_) {
    throw std::out_of_range("CumulativeHistogram::partialSum(): k is out of range.");
  }
  if (k == num_elements_ - 1) {
    return total_sum_;
  }
  T result {};
  // Tree representing the elements [0; N).
  const std::span<const T> nodes = std::span<const T>{data_}.subspan(num_elements_);
  TreeView<false> tree(nodes, 0, num_elements_ - 1);
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
constexpr T CumulativeHistogram<T>::totalSum() const noexcept {
  return total_sum_;
}

}  // namespace CumulativeHistogram_NS
