#include "cuda_half.h"

#include <memory>
#include <string>
#include <utility>
#include "exr.h"
#include "image.h"
#include "timer.h"

#include "blur.cu"  // Include source directly for now

struct CudaFreeFunctor {
  void operator()(void* p) noexcept { cudaCheckError(cudaFree(p)); }
};

template <class T>
std::unique_ptr<T, CudaFreeFunctor> cuda_malloc_unique(size_t size) {
  T* device_image;
  cudaCheckError(cudaMalloc(&device_image, size));
  return std::unique_ptr<T, CudaFreeFunctor>(device_image);
}

template <class T>
auto alloc_and_copy(T* host_image, image_dims dims) {
  using deviceT = typename std::remove_const<T>::type;
  auto size = allocated_bytes(dims);
  auto device_image = cuda_malloc_unique<deviceT>(size);
  cudaCheckError(
      cudaMemcpy(device_image.get(), host_image, size, cudaMemcpyDefault));
  return device_image;
}

void copy_image(void* dest, const void* source, image_dims dims) {
  cudaCheckError(
      cudaMemcpy(dest, source, allocated_bytes(dims), cudaMemcpyDefault));
}

int main(int argc, char** argv) {
  auto [pixels, dims] = read_exr(argv[1]);

  auto device_pixels = alloc_and_copy(pixels.get(), dims);

  // Copy result back to host and write
  copy_image(pixels.get(), device_pixels.get(), dims);
  write_exr(argv[2], dims, pixels.get());
  return 0;
}
