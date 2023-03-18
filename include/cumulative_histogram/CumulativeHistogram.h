#include <bit>
#include <memory>
#include <stdexcept>
#include <span>
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
  // TODO: time complexity?
  explicit CumulativeHistogram(std::size_t num_elements);

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

 private:
  static std::size_t countNodesInTree(std::size_t num_elements);

  std::vector<T> data_;
  // Tree data.
  // The first element is the root. The root stores the sum of elements from its left branch.
  // The left branch is a tree representing elements [0; (N-1)/2].
  // The right branch is a tree representing elements [(N+1)/2; N).
  //
  // The tree is an implicit data structure:
  //   * We know the total number of nodes in the tree from the formula in countNodesInTree().
  //   * The tree is balanced: for any node the numbers of nodes in the left and right branches
  //     differ at most by 1.
  //   * Therefore, we can store it in a plain array like this:
  // Number of elements = 11; total number of nodes = 6
  //         _ n0 __
  //        /       \
  //       n1        n4
  //     /    \      /
  //   n2     n3     n5
  // .......................implicit nodes........
  //  /  \    / \    /  \    \
  // x0  x2  x3  x5 x6  x8   x9
  //  \       \      \        \
  //   x1      x4    x7      x10
  //
  // Nodes:
  // +-------------------+----------+---------+---------+----------+---------+
  // |    n0             |    n1    |    n2   |    n3   |    n4    |    n5   |
  // +-------------------+----------+---------+---------+----------+---------+
  // | x0+x1+x2+x3+x4+x5 | x0+x1+x2 | x0 + x1 | x3 + x4 | x6+x7+x8 | x6 + x7 |
  // +-------------------+----------+---------+---------+----------+---------+
  // Any partial sum can be computed by going from the root of the tree to the leaf:
  // +------------------------------------------------------------------------------------------+
  // |  s0  |  s1  |  s2  |  s3   |  s4   |  s5  |  s6   |  s7   |  s8   |  s9      |  s10      |
  // +------------------------------------------------------------------------------------------+
  // |  x0  |  n2  |  n1  | n1+x3 | n1+n3 |  n0  | n0+x6 | n0+n5 | n0+n4 | n0+n4+x9 | total_sum |
  // +------------------------------------------------------------------------------------------+
  std::vector<T> nodes_;

  T total_sum_ = {};
};

template<class T>
std::size_t CumulativeHistogram<T>::countNodesInTree(std::size_t num_elements) {
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

template<class T>
CumulativeHistogram<T>::CumulativeHistogram(std::size_t num_elements):
  data_(num_elements),
  nodes_(countNodesInTree(num_elements))
{}

template<class T>
constexpr std::size_t CumulativeHistogram<T>::numElements() const noexcept {
  return data_.size();
}

template<class T>
constexpr std::span<const T> CumulativeHistogram<T>::elements() const noexcept {
  return std::span<const T>{data_};
}

template<class T>
const T& CumulativeHistogram<T>::element(std::size_t k) const {
  return data_.at(k);
}

template<class T>
void CumulativeHistogram<T>::increment(std::size_t k, const T& value) {
  const std::size_t n = data_.size();
  if (k >= n) {
    throw std::out_of_range("CumulativeHistogram::increment(): k is out of range.");
  }
  std::size_t first = 0;     // inclusive
  std::size_t last = n - 1;  // inclusive
  // Indices of the nodes representing the subtree for elements [first; last].
  auto root = nodes_.begin();  // iterator to the root node of the current subtree.
  std::size_t num_nodes = nodes_.size();  // the number of nodes in the current subtree.
  while (num_nodes != 0) {
    // Elements [first; middle] are in the left branch.
    // Elements [middle+1; last] are in the right branch.
    const std::size_t middle = first + (last - first) / 2;
    // The left subtree has ceil((num_nodes-1)/2) = floor(num_nodes/2) nodes.
    const std::size_t num_nodes_left = num_nodes / 2;
    if (k <= middle) {
      // Increment the root: the root of a subtree stores the sum elements [first; last].
      *root += value;
      // Switch to the left tree:
      last = middle;
      ++root;
      num_nodes = num_nodes_left;
    } else {
      // Not incrementing the root, because element k is not in the left branch.
      // Switch to the right tree:
      first = middle + 1;
      // The right subtree has floor((num_nodes-1)/2) nodes.
      const std::size_t num_nodes_right = (num_nodes - 1) / 2;
      root += (1 + num_nodes_left);
      num_nodes = num_nodes_right;
    }
  }
  // Update the element itself.
  data_[k] += value;
  // Update the total sum.
  total_sum_ += value;
}

template<class T>
T CumulativeHistogram<T>::partialSum(std::size_t k) const {
  const std::size_t n = data_.size();
  if (k >= n) {
    throw std::out_of_range("CumulativeHistogram::increment(): k is out of range.");
  }
  if (k == n - 1) {
    return total_sum_;
  }
  std::size_t first = 0;     // inclusive
  std::size_t last = n - 1;  // inclusive
  T result {};
  // Indices of the nodes representing the subtree for elements [first; last].
  auto root = nodes_.begin(); // iterator to the root node of the current subtree.
  std::size_t num_nodes = nodes_.size();  // the number of nodes in the current subtree.
  while (num_nodes != 0) {
    const std::size_t middle = first + (last - first) / 2;
    if (k == middle) {
      // We are in luck - the left subtree represents elements [first; middle], so
      // there's no need to traverse any deeper.
      return result + *root;
    }
    // The left subtree has ceil((num_nodes-1)/2) = floor(num_nodes/2) nodes.
    const std::size_t num_nodes_left = num_nodes / 2;
    if (k < middle) {
      // Switch to the left subtree:
      last = middle;
      ++root;
      num_nodes = num_nodes_left;
    } else {  // k > middle
      // Add the sum of elements from [first; middle].
      result += *root;
      // Switch to the right subtree:
      first = middle + 1;
      // The right subtree has floor((num_nodes-1)/2) nodes.
      const std::size_t num_nodes_right = (num_nodes - 1) / 2;
      root += (1 + num_nodes_left);
      num_nodes = num_nodes_right;
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
