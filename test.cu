#include "cuda_half.h"

#include <nppi_filtering_functions.h>
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

static void npp_blur(float4* dest, const float4* source, float4* temp,
                     image_dims dims, int radius, int n_passes) {
  auto* from = reinterpret_cast<Npp32f*>(temp);
  auto* to = reinterpret_cast<Npp32f*>(dest);

  NppiSize size = {static_cast<int>(dims.width), static_cast<int>(dims.height)};

  int remaining = radius;
  for (int i = 0; i < n_passes; i++) {
    int this_radius = remaining / (n_passes - i);
    remaining -= this_radius;
    int width = 2 * this_radius + 1;

    // On the first pass, read from the source image
    const auto* s = (i == 0) ? reinterpret_cast<const Npp32f*>(source) : from;

    size_t line_bytes = stride_bytes(dims);
    nppiFilterBoxBorder_32f_C4R(s, line_bytes,               //
                                size, {0, 0},                //
                                to, line_bytes, size,        //
                                {width, width},              // filter size
                                {this_radius, this_radius},  // center
                                NPP_BORDER_REPLICATE);
    std::swap(to, from);
  }

  // FIXME: this will only work for odd numbers of passes
}

const bool do_crop = false;
const bool do_outputs = false;
const bool do_column_split = false;
const bool do_npp = false;
const bool do_direct = false;

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

  if (do_outputs) {
    for (int outputs_v = 1; outputs_v <= 3; outputs_v++) {
      for (int outputs_h = 1; outputs_h <= 3; outputs_h++) {
        timeit("outputs/thread " + std::to_string(outputs_v) +
                   std::to_string(outputs_h),
               [&]() {
                 smooth_blur(dest.get(), source.get(), temp.get(), dims, radius,
                             n_passes, outputs_v, outputs_h, 1, 2);
               });
      }
    }
  }

  if (do_column_split) {
    for (int threads_v = 1; threads_v <= 4; threads_v++) {
      for (int threads_h = 1; threads_h <= 4; threads_h++) {
        timeit("threads/column " + std::to_string(threads_v) +
                   std::to_string(threads_h),
               [&]() {
                 smooth_blur(dest.get(), source.get(), temp.get(), dims, radius,
                             n_passes, 1, 1, threads_v, threads_h);
               });
      }
    }
  }

  if (do_npp) {
    timeit("npp blur", [&]() {
      npp_blur(dest.get(), source.get(), temp.get(), dims, radius, n_passes);
    });
  }

  if (do_direct) {
    for (int outputs = 2; outputs <= 3; outputs++) {
      timeit("direct blur x" + std::to_string(outputs), [&]() {
        direct_blur_horizontal(dest.get(), source.get(), temp.get(), dims,
                               radius, n_passes, outputs);
      });
      timeit("direct blur y" + std::to_string(outputs), [&]() {
        direct_blur_vertical(dest.get(), source.get(), temp.get(), dims, radius,
                             n_passes, outputs);
      });
      timeit("no transpose" + std::to_string(outputs), [&]() {
        direct_blur_no_transpose(dest.get(), source.get(), temp.get(), dims,
                                 radius, n_passes, outputs);
      });
    }
  }

  // Current fastest configuration (at 1920x1080, radius 10)
  timeit("fastest blur", [&]() {
    smooth_blur(dest.get(), source.get(), temp.get(), dims, radius, n_passes, 3,
                3, 1, 2);
  });

  // Copy result back to host and write
  copy_image(pixels.get(), dest.get(), dims);
  write_exr(argv[2], dims, pixels.get());

  return 0;
}
