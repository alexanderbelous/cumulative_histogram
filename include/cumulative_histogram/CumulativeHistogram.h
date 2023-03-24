#include <bit>
#include <numeric>
#include <stdexcept>
#include <span>
#include <type_traits>
#include <vector>

namespace CumulativeHistogram_NS {

// This class allows to efficiently compute partial sums for a dynamic array of elements
// by offering a compromise between 2 naive solutions:
// +------------------------------------+-------------------+-------------+-------+
// | Solution                           | Modify an element | Partial sum | Space |
// +------------------------------------+-------------------+-------------+-------+
// | 1. Store the elements in an array  | O(1)              | O(N)        | O(N)  |
// +------------------------------------+-------------------+-------------+-------+
// | 2. Store the partial sums in array | O(N)              | O(1)        | O(N)  |
// +------------------------------------+-------------------+-------------+-------+
// | 3. CumulativeHistogram             | O(logN)           | O(logN)     | O(N)  |
// +------------------------------------+-------------------+-------------+-------+
//
// Unlike a Fenwick tree, this class allows adding and removing elements.
// The memory overhead is approximately 50% (i.e. the class stores roughly N/2 extra counters).
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

  // Returns true if the number of elements is greater than 0, false otherwise.
  // Time complexity: O(1).
  constexpr bool empty() const noexcept;

  // Returns the number of elements in the histogram.
  constexpr std::size_t size() const noexcept;

  // Returns the number of elements that can be held in currently allocated storage.
  // Time complexity: O(1).
  constexpr std::size_t capacity() const noexcept;

  // Reserves memory for a histogram capable of storing the specified number of elements.
  // The values of existing elements remain unchanged.
  // Time complexity: O(N), where N is the current number of elements.
  void reserve(std::size_t num_elements);

  // Erases all elements
  // The capacity remains unchanged.
  void clear();

  // TODO
  void resize(std::size_t num_elements);

  // Add a zero-initialized element to the end.
  // Time complexity: amortized O(1).
  void push_back();

  // Add an element to the end.
  // Time complexity: amortized O(logN).
  void push_back(const T& value);

  // Removes the last element.
  // Throws std::logic_error if this->empty().
  // Time complexity: O(logN).
  //
  // TODO:
  //   AFAIU, it should be possible to make it O(1) at the cost of
  // making push_back() a bit slower (though still having O(logN) time complexity).
  //   When the rightmost element is removed, only the total sum actually needs to be updated.
  // Yes, other nodes are affected, but the affected nodes will not be accesed anymore
  // (the node is affected if it contains x[i]; if we remove x[i] then no valid call to
  // partialSum() will actually add values from the nodes that contain x[i]).
  //   However, this means that these nodes will store garbage data; if we call push_back()
  // afterwards, we'll need to ensure that it zero-initializes these nodes before incrementing
  // them. This can still be done in O(logN) - e.g., push_back() can initialize the
  // nodes that contain the new element, and there's at most O(log) of them.
  //   Then again, we might want to do this in push_back() anyway for 2 reasons:
  //     1. To avoid zeroing out newly allocated memory. reserve(M) will have O(M) time coplexity
  //        if we zero-initialize, but only O(N) if we don't.
  //     2. To reduce the rounding errors when T is a floating-point type.
  void pop_back();

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
  const T& element(std::size_t k) const;

  // Increment the specified element.
  // Throws std::out_of_range if k >= size().
  // Time complexity: O(log2(N)).
  void increment(std::size_t k, const T& value = 1);

  // Returns the partial sum of the first K elements.
  // Throws std::out_of_range if k >= size().
  // Time complexity: O(log2(N)).
  T partialSum(std::size_t k) const;

  // Returns the total sum of all elements.
  // Throws std::logic_error if this->empty().
  // Time complexity: O(1).
  constexpr T totalSum() const;

  // Find the first element k, for which partialSum(k) is not less than the specified value.
  // Time complexity: O(log2(N)).
  // TODO: implement.

 private:
  // Returns the total number of nodes in the auxiliary tree for
  // CumulativeHistogram with the specified number of elements.
  static constexpr std::size_t countNodesInTree(std::size_t num_elements) noexcept;

  // Returns the maximum number of elements that can represented by the current tree.
  //   num_elements <= capacityCurrent() <= capacity_
  constexpr std::size_t capacityCurrent() const noexcept;

  // Returns the number of nodes in the current tree (not including the element storing the total sum).
  constexpr std::size_t numNodesCurrent() const noexcept;

  // Rebuilds the tree data.
  // Time complexity: O(N).
  void rebuildTree() noexcept;

  template<bool Mutable>
  class TreeView;

  T buildTreeImpl(const TreeView<true>& tree) noexcept;

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
  std::size_t num_elements_ = 0;
  // Current capacity Nmax.
  std::size_t capacity_ = 0;
  // Index of the root node.
  // If data_ is not empty, then 0 < N <= Nmax <= root_idx - 1.
  // data_[root_idx - 1] always stores the total sum of elements [0; N).
  // TODO: consider computing this number on the fly. Basically:
  //   if (num_elements > ceil(capacity_/2))
  //     root_idx_ = capacity_ + 1;
  //   else if (num_elements > ceil(ceil(capacity_/2)/2))
  //     root_idx = capacity + 2;
  //   else if (...)
  std::size_t root_idx_ = 1;
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

template<class T>
constexpr std::size_t CumulativeHistogram<T>::capacityCurrent() const noexcept {
  // 1. Determine the depth of the current root node in the "full" tree.
  // Note that there aren't actually any nodes if capacity_ < 3. In that case root_idx_
  // should still equal capacity_ + 1 (even though data[root_idx_] is out of range).
  // level will be 0, and this function will simply return capacity_.
  const std::size_t level = root_idx_ - capacity_ - 1;
  // 2. f(0) = Nmax
  //    f(1) = ceil(f(0)/2) = (Nmax+1)/2
  //    f(2) = ceil(f(1)/2) = ((Nmax+1)/2 + 1)/2 = (Nmax+3)/4 = ceil(Nmax/4)
  //    f(3) = ceil(f(2)/2) = (Nmax+7)/8 = ceil(Nmax/8)
  //    f(N) = ceil(Nmax/2^N) = (Nmax + 2^N - 1) / 2^N
  return (capacity_ + (static_cast<std::size_t>(1) << level) - 1) >> level;
}

template<class T>
constexpr std::size_t CumulativeHistogram<T>::numNodesCurrent() const noexcept {
  return countNodesInTree(capacityCurrent());
}

// TODO: replace recursion with a loop.
template<class T>
T CumulativeHistogram<T>::buildTreeImpl(const TreeView<true>& tree) noexcept {
  // Every node represents at least 3 elements - we don't store smaller nodes.
  // Thus, any non-empty tree has elements in both left and right subtrees, even
  // if these subtrees don't actually have nodes.
  if (tree.empty()) {
    const auto first = data_.begin() + tree.elementFirst();
    const auto last = data_.begin() + tree.elementLast() + 1;
    return std::accumulate(first, last, T{});
  }
  const T total_sum_left = buildTreeImpl(tree.leftChild());
  const T total_sum_right = buildTreeImpl(tree.rightChild());
  tree.root() = total_sum_left;
  return total_sum_left + total_sum_right;
}

template<class T>
void CumulativeHistogram<T>::rebuildTree() noexcept {
  const std::span<T> nodes = std::span<T>{data_}.subspan(root_idx_, numNodesCurrent());
  TreeView<true> tree(nodes, 0, capacityCurrent() - 1);
  data_[root_idx_ - 1] = buildTreeImpl(tree);
  // TODO: zero-initialize reserved elements?
}

template<class T>
CumulativeHistogram<T>::CumulativeHistogram(std::size_t num_elements):
  data_(num_elements ? num_elements + 1 + countNodesInTree(num_elements) : 0),
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
  data_.resize(num_elements_ + 1 + countNodesInTree(num_elements_));
  rebuildTree();
}

// TODO: if Iter allows multipass, we should only do 1 memory allocation.
template<class T>
template<class Iter>
CumulativeHistogram<T>::CumulativeHistogram(Iter first, Iter last):
  CumulativeHistogram(std::vector<T>{first, last})
{}

template<class T>
constexpr bool CumulativeHistogram<T>::empty() const noexcept {
  return num_elements_ == 0;
}

template<class T>
constexpr std::size_t CumulativeHistogram<T>::size() const noexcept {
  return num_elements_;
}

template<class T>
constexpr std::size_t CumulativeHistogram<T>::capacity() const noexcept {
  return capacity_;
}

template<class T>
void CumulativeHistogram<T>::setZero() {
  std::fill(data_, T{});
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
  const std::size_t capacity_current = capacityCurrent();
  // The number of nodes in the current tree.
  const std::size_t nodes_current = countNodesInTree(capacity_current);
  --num_elements_;
  if (nodes_current > 0) {
    const std::size_t num_elements_in_left_subtree = (capacity_current + 1) / 2;
    // If all "real" elements are now in the left subtree, declare the left subtree as the new tree.
    if (num_elements_ <= num_elements_in_left_subtree) {
      ++root_idx_;
    }
  }
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
  return data_[k];
}

template<class T>
void CumulativeHistogram<T>::increment(std::size_t k, const T& value) {
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
T CumulativeHistogram<T>::partialSum(std::size_t k) const {
  if (k >= num_elements_) {
    throw std::out_of_range("CumulativeHistogram::partialSum(): k is out of range.");
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

}  // namespace CumulativeHistogram_NS
