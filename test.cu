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

// Reduce width but keep stride unchanged, to test handling of padded images.
// Fill padding with infinite values, which should make it obvious if these
// pixels are accidentally accessed.
void crop_image(float4* image, image_dims& dims) {
  dims.width -= 9;
  auto inf = std::numeric_limits<float>::infinity();

  for (size_t y = 0; y < dims.height; y++) {
    for (size_t x = dims.width; x < dims.stride_pixels; x++) {
      image[pixel_index(dims, x, y)] = {inf, inf, inf, inf};
    }
  }
}

const bool do_crop = false;
const bool do_texture = false;

int main(int argc, char** argv) {
  int radius = (argc > 3) ? std::stoi(argv[3]) : 5;
  int n_passes = (argc > 4) ? std::stoi(argv[4]) : 3;

  auto [pixels, _dims] = read_exr(argv[1]);
  // Work around stupid "structured binding can't be captured" issue
  auto dims = _dims;

  if (do_crop)
    crop_image(pixels.get(), dims);

  auto source = alloc_and_copy(pixels.get(), dims);
  auto dest = cuda_malloc_unique<float4>(allocated_bytes(dims));
  auto temp = cuda_malloc_unique<float4>(allocated_bytes(dims));

  timeit("smooth blur", [&]() {
    smooth_blur(dest.get(), source.get(), temp.get(), dims, radius, n_passes);
  });

  if (do_texture) {
    timeit("texture blur", [&]() {
      smooth_blur_texture(dest.get(), source.get(), temp.get(), dims, radius,
                          n_passes);
    });
  }

  // Copy result back to host and write
  copy_image(pixels.get(), dest.get(), dims);
  write_exr(argv[2], dims, pixels.get());

  return 0;
}
