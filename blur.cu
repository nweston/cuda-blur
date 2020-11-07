#include "cuda_runtime_api.h"

#define cudaCheckError(code)                                         \
  {                                                                  \
    if ((code) != cudaSuccess) {                                     \
      handleCudaError(__FILE__, __LINE__, cudaGetErrorString(code)); \
    }                                                                \
  }

void handleCudaError(const char* file, int line, const char* error) {
  std::cerr << "CUDA failure " << file << ":" << line << " " << error << "\n";
}
