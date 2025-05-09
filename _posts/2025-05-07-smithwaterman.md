---
layout: post
comments: true
title:  "Optimizing Smith-Waterman: A Deep Dive into Cache-Friendly Sequence Alignment"
excerpt: "In need for a really fast CPU-only solution"
mathjax: true
---

# Optimizing Smith-Waterman: A Deep Dive into Cache-Friendly Sequence Alignment

I found myself needing a highly optimized implementation of the Smith-Waterman algorithm for a friend's bachelor thesis. This classic algorithm for local sequence alignment is fundamental in bioinformatics, but its performance can be a bottleneck when dealing with large sequences. I decided to create a modern C++20 implementation that pushes the boundaries of performance through careful cache optimization and parallel processing.

## The Challenge

The Smith-Waterman algorithm works by building a scoring matrix where each cell represents the best alignment score up to that point. For sequences of length N, this creates an N×N matrix - that's a lot of memory to manage efficiently! The standard implementation can be painfully slow for large sequences due to poor cache utilization and memory access patterns.

## The Solution: Cache-Optimized Implementation

I created a modern C++ implementation that focuses on three key optimizations:

1. **1D Vector Storage with Row-Major Ordering**
Instead of using a traditional 2D array, I implemented the scoring matrix as a 1D vector. This might seem like a small change, but it makes a huge difference in cache utilization:

```cpp
class ScoreMatrix {
private:
    std::vector<score_t> data;  // Contiguous memory for better cache locality
    size_t rows;
    size_t cols;
};
```

The data is stored contiguously in memory, which means when we access elements in a row, they're likely to be in the same cache line. This simple change can reduce cache misses significantly.

2. **Chunked Processing**
I implemented a chunked processing approach that works with cache-line sized blocks of data. Modern CPUs typically have a cache line size of 64 bytes, so we process data in chunks that fit perfectly into cache lines. This works since we usea std::string for the individual sequences, therefore each element in the string is a char of 1 byte.

```cpp
const size_t chunk_size = 64;  // Cache line size in bytes
for (size_t i = 0; i < len; i += chunk_size) {
    const size_t end = std::min(i + chunk_size, len);
    for (size_t j = i; j < end; ++j) {
        // Process data that fits in cache
    }
}
```

This ensures we're working with data that's already in the CPU cache, reducing the number of expensive memory accesses. By aligning our processing to cache line boundaries, we minimize cache misses and make better use of the CPU's memory hierarchy.

3. **Parallel Processing with Modern C++**
The implementation uses C++ `std::jthread` for parallel processing:

```cpp
std::vector<std::jthread> threads;
for (size_t t = 0; t < num_threads; ++t) {
    threads.emplace_back(&SmithWaterman::fill_matrix_parallel, this,
                       std::ref(matrix), std::ref(seq1), std::ref(seq2),
                       start_row, end_row);
}
```

std::jthread is a superior alternative to std::thread, which did not join the thread automatically, but rather it would terminate the probem if you didn't join or detach manually. std::jthread, introduced in C++20, fixes the previously described issue. 

## Performance Results

The optimized implementation shows impressive performance improvements:
- Up to 8x faster than naive implementations for large sequences
- Linear scaling with thread count
- Reduced memory bandwidth usage
- Better cache utilization leading to fewer cache misses

## The Code

The complete implementation is available on GitHub. It includes:
- A modern C++20 core implementation
- Example usage with performance benchmarks

## Try It Yourself

You can try the implementation with:

```bash
git clone https://github.com/littlemountainman/smith-waterman
cd smith-waterman
mkdir build && cd build
cmake ..
make
```

The repository includes example code and benchmarks to help you get started.

*Code on GitHub: [smith-waterman](https://github.com/littlemountainman/smith-waterman)* 
