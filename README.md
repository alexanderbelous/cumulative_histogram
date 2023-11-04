# cumulative_histogram
Fast cumulative histogram (C++)

The class `CumulativeHistogram` in this library allows to efficiently compute prefix sums for a
dynamic array of elements. The 2 naive solutions (storing the elements as a plain array or
storing an array of partial sums) have very good time complexity for some operations, but
perform poorly for the others. This class offers a compromise between them.

Another data structure for this problem, called Fenwick tree, also offers good time complexity for both updating
individual elements and computing prefix sums, but not for adding/removing elements.

| Data structure       | getElement() | updateElement() | prefixSum() | lowerBound() | pushBack() | popBack() | Space complexity |
|----------------------|--------------|-----------------|-------------|--------------|------------|-----------|------------------|
| Array of elements    | O(1)         | O(1)            | O(N)        | O(N)         | O(1)+      | O(1)      | O(N)             |
| Array of prefix sums | O(1)         | O(N)            | O(1)        | O(logN)      | O(1)+      | O(1)      | O(N)             |
| Fenwick tree         | O(logN)      | O(logN)         | O(logN)     | O(logN)      | N/A        | N/A       | O(N)             |
| CumulativeHistogram  | O(1)         | O(logN)         | O(logN)     | O(logN)      | O(1)+      | O(1)      | O(N)             |

# Basic usage
```cpp
// Construct a histogram for 100 elements, and initialize them with 1.
CumulaitveHistogram<int> histogram(100, 1);

// Access elements:
histogram.element(0);   // returns 1
histogram.element(10);  // returns 1
histogram.element(99);  // returns 1;
std::span<const int> elements = histogram.elements();  // Elements are stored contiguously.

// Compute prefix sums:
histogram.prefixSum(0);   // returns 1
histogram.prefixSum(10);  // returns 11
histogram.prefixSum(99);  // returns 100
histogram.totalSum();     // returns 100

// Update an element:
histogram.increment(10, 7);  // increment the 10th element by 7.

// Append a new element (initalized with 5) to the end:
histogram.push_back(5);

// Remove the last element:
histogram.pop_back();

// Find the index k such that histogram.prefixSum(k) >= 42:
auto [iter, prefix_sum] = histogram.lowerBound(42);            // &(*iter) == &(elements[34]), prefix_sum == 42
const std::size_t k = std::distance(histogram.begin(), iter);  // k == 34
```
