# cumulative_histogram
Fast cumulative histogram (C++)

This library provides a container `CumulativeHistogram` that allows to efficiently compute prefix sums for
a dynamic array of elements. The 2 naive solutions (storing the elements as a plain array or storing an array
of partial sums) have very good time complexity for some operations, but perform poorly for the others. This
class offers a compromise between them.

Another data structure for this problem, called Fenwick tree, also offers good time complexity for both updating
individual elements and computing prefix sums, but not for adding/removing elements.

| Data structure       | getElement() | updateElement() | prefixSum() | lowerBound() | pushBack() | popBack() | Space complexity |
|----------------------|--------------|-----------------|-------------|--------------|------------|-----------|------------------|
| Array of elements    | O(1)         | O(1)            | O(N)        | O(N)         | O(1)+      | O(1)      | O(N)             |
| Array of prefix sums | O(1)         | O(N)            | O(1)        | O(logN)      | O(1)+      | O(1)      | O(N)             |
| Fenwick tree         | O(logN)      | O(logN)         | O(logN)     | O(logN)      | N/A        | N/A       | O(N)             |
| CumulativeHistogram  | O(1)         | O(logN)         | O(logN)     | O(logN)      | O(1)+      | O(1)      | O(N)             |

## Basic usage
```cpp
// Construct a histogram for 10 elements, and initialize them with 1.
CumulaitveHistogram<int> histogram(10, 1);

// Access elements:
histogram.element(0);  // returns 1
histogram.element(1);  // returns 1
histogram.element(9);  // returns 1;
std::span<const int> elements = histogram.elements();  // Elements are stored contiguously.

// Compute prefix sums:
histogram.prefixSum(0);   // returns 1
histogram.prefixSum(1);   // returns 2
histogram.prefixSum(9);   // returns 10
histogram.totalSum();     // returns 10

// Update an element:
histogram.increment(1, 7);  // increment the 1st element by 7. element(1) will now return 8.

// Append a new element (initalized with 5) to the end:
histogram.push_back(5);

// Remove the last element:
histogram.pop_back();

// Find the first index k such that histogram.prefixSum(k) >= 10:
auto [iter, prefix_sum] = histogram.lowerBound(10);            // iter points to element(2), prefix_sum == 10
const std::size_t k = std::distance(histogram.begin(), iter);  // k == 2
```
