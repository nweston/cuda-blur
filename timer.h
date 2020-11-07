//  -*- mode: c++ -*-

#include <chrono>
#include <iostream>
#include "cuda_runtime.h"

// Time execution of a function.
// Calls cudaDeviceSynchronize() before stopping the timer to ensure any
// running kernels are finished.
template <class T>
void timeit(const std::string &name, T function,
            std::ostream &out = std::cout) {
  auto start = std::chrono::high_resolution_clock::now();
  function();
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  // Duration in miliseconds
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  out << name << ": " << elapsed.count() << " ms\n";
}