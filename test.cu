#include "cuda_half.h"

#include <nppi_filtering_functions.h>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "exr.h"
#include "image.h"
#include "timer.h"

#include "blur.cu"  // Include source directly for now

// ===== half4 type =====
struct half4 {
  half x;
  half y;
  half z;
  half w;
};
template <>
struct temp_type<half4> {
  using type = float4;
};
template <>
float4 to_temp(const half4& a) {
  return {a.x, a.y, a.z, a.w};
}
template <>
half4 from_temp(const float4& a) {
  return {a.x, a.y, a.z, a.w};
}
// ==========

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

auto inf = std::numeric_limits<float>::infinity();
template <class T>
static T infinity();
template <>
float4 infinity() {
  return {inf, inf, inf, inf};
}
template <>
float2 infinity() {
  return {inf, inf};
}
template <>
float infinity() {
  return inf;
}
template <>
half2 infinity() {
  return {inf, inf};
}
template <>
half4 infinity() {
  return {inf, inf, inf, inf};
}

// Reduce width but keep stride unchanged, to test handling of padded images.
// Fill padding with infinite values, which should make it obvious if these
// pixels are accidentally accessed.
template <class T>
static void crop_image(T* image, image_dims& dims) {
  dims.width -= 9;

  for (size_t y = 0; y < dims.height; y++) {
    for (size_t x = dims.width; x < dims.stride_pixels; x++) {
      image[pixel_index(dims, x, y)] = infinity<T>();
    }
  }
}

const int CUDA_ALIGN = 256;
static image_dims convert_dims(image_dims dims, int channel_count,
                               size_t sizeof_channel) {
  image_dims new_dims = dims;
  new_dims.channel_count = channel_count;
  new_dims.sizeof_channel = sizeof_channel;
  size_t stride =
      (stride_bytes(new_dims) + CUDA_ALIGN - 1) / CUDA_ALIGN * CUDA_ALIGN;
  new_dims.stride_pixels = stride / pixel_size(new_dims);

  return new_dims;
}

template <class DestT, class SourceT>
static DestT convert(const SourceT& source) {
  return {source.x, source.y, source.z, source.w};
}
template <>
float4 convert(const float2& source) {
  return {source.x, source.y, 0, 1};
}
template <>
float2 convert(const float4& source) {
  return {source.x, source.y};
}
template <>
float4 convert(const float& source) {
  return {source, 0, 0, 1};
}
template <>
float convert(const float4& source) {
  return source.x;
}
template <>
half2 convert(const float4& source) {
  return {source.x, source.y};
}
template <>
float4 convert(const half2& source) {
  return {source.x, source.y, 0, 1};
}

template <class DestT, class SourceT>
static void convert_image(DestT* dest, SourceT* source, image_dims dest_dims,
                          image_dims source_dims) {
  for (size_t y = 0; y < source_dims.height; y++) {
    for (size_t x = 0; x < source_dims.width; x++) {
      dest[pixel_index(dest_dims, x, y)] =
          convert<DestT>(source[pixel_index(source_dims, x, y)]);
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

static float diff(const float4& a, const float4& b) {
  return std::abs(std::max({a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}));
}

template <class T>
static void run_checks_internal(T* pixels, image_dims dims,
                                const std::string& format,
                                std::vector<std::string>& failures) {
  auto source = alloc_and_copy(pixels, dims);
  auto dest = cuda_malloc_unique<T>(allocated_bytes(dims));
  auto temp = cuda_malloc_unique<T>(allocated_bytes(dims));

  std::vector<int> radii{1, 3, 5, 10, 20, 50, 100};
  auto baseline = std::make_unique<float4[]>(allocated_pixels(dims));
  auto result = std::make_unique<float4[]>(allocated_pixels(dims));

  const int n_passes = 3;
  for (int crop = 0; crop <= 1; crop++) {
    if (crop) {
      crop_image(pixels, dims);
    }

    for (auto radius : radii) {
      auto check_result = [&](const std::string& name,
                              float threshold = 0.0001) {
        copy_image(result.get(), dest.get(), dims);
        float max_diff = 0;
        for (size_t y = 0; y < dims.height; y++) {
          for (size_t x = 0; x < dims.width; x++) {
            auto index = pixel_index(dims, x, y);
            max_diff = std::max(
                max_diff, diff(baseline.get()[index], result.get()[index]));
          }
        }
        if (max_diff > threshold) {
          std::cout << "x" << std::flush;
          failures.push_back(name + "[" + format + "]" +
                             (crop ? " (cropped) " : " ") + "radius " +
                             std::to_string(radius) +

                             " failed: max diff " + std::to_string(max_diff));
        } else {
          std::cout << "." << std::flush;
        }
      };

      // Use iterated box blur, one thread per column, as the baseline.
      smooth_blur(dest.get(), source.get(), temp.get(), dims, radius, n_passes);
      copy_image(baseline.get(), dest.get(), dims);

      // Multiple outputs per thread
      for (int outputs_v = 2; outputs_v <= 3; outputs_v++) {
        for (int outputs_h = 2; outputs_h <= 3; outputs_h++) {
          smooth_blur(dest.get(), source.get(), temp.get(), dims, radius,
                      n_passes, outputs_v, outputs_h, 1, 1);
          check_result("outputs " + std::to_string(outputs_v) +
                       std::to_string(outputs_h));
        }
      }

      // Column splitting
      for (int threads_v = 2; threads_v <= 4; threads_v++) {
        for (int threads_h = 2; threads_h <= 4; threads_h++) {
          smooth_blur(dest.get(), source.get(), temp.get(), dims, radius,
                      n_passes, 1, 1, threads_v, threads_h);
          check_result("columns " + std::to_string(threads_v) +
                       std::to_string(threads_h));
        }
      }

      // Direct box filter
      direct_blur_no_transpose(dest.get(), source.get(), temp.get(), dims,
                               radius, n_passes, 2);
      check_result("direct");

      // Direct Gaussian blur
      if (radius <= MAX_PRECOMPUTED_RADIUS) {
        precomputed_gaussian_blur(dest.get(), source.get(), temp.get(), dims,
                                  radius, 2);
        // Large threshold for now, due to edge artifacts
        check_result("gaussian");
      }
    }
  }
}

static int run_checks(float4* pixels, image_dims dims) {
  std::vector<std::string> failures;

  run_checks_internal(pixels, dims, "float4", failures);

  // float2
  {
    auto dims2 = convert_dims(dims, 2, sizeof(float));
    auto pixels2 = std::make_unique<float2[]>(allocated_pixels(dims2));
    convert_image(pixels2.get(), pixels, dims2, dims);
    run_checks_internal(pixels2.get(), dims2, "float2", failures);
  }

  // float
  {
    auto dims2 = convert_dims(dims, 1, sizeof(float));
    auto pixels2 = std::make_unique<float[]>(allocated_pixels(dims2));
    convert_image(pixels2.get(), pixels, dims2, dims);
    run_checks_internal(pixels2.get(), dims2, "float", failures);
  }

  // half4
  {
    auto dims2 = convert_dims(dims, 4, sizeof(half));
    auto pixels2 = std::make_unique<half4[]>(allocated_pixels(dims2));
    convert_image(pixels2.get(), pixels, dims2, dims);
    run_checks_internal(pixels2.get(), dims2, "half4", failures);
  }

  // half2
  {
    auto dims2 = convert_dims(dims, 2, sizeof(half));
    auto pixels2 = std::make_unique<half2[]>(allocated_pixels(dims2));
    convert_image(pixels2.get(), pixels, dims2, dims);
    run_checks_internal(pixels2.get(), dims2, "half2", failures);
  }
  std::cout << "\n";

  if (failures.empty()) {
    std::cout << "OK!\n";
    return 0;
  } else {
    for (auto& n : failures) {
      std::cout << n << "\n";
    }
    return failures.size();
  }
}

template <class T>
void convert_and_test(float4* pixels, image_dims dims, int radius, int n_passes,
                      int channel_count, size_t sizeof_channel,
                      const std::string& label) {
  auto dims2 = convert_dims(dims, channel_count, sizeof_channel);
  auto pixels2 = std::make_unique<T[]>(allocated_pixels(dims2));
  convert_image(pixels2.get(), pixels, dims2, dims);

  auto source2 = alloc_and_copy(pixels2.get(), dims2);
  auto dest2 = cuda_malloc_unique<T>(allocated_bytes(dims2));
  auto temp2 = cuda_malloc_unique<T>(allocated_bytes(dims2));

  timeit("fastest blur " + label, [&]() {
    smooth_blur(dest2.get(), source2.get(), temp2.get(), dims, radius, n_passes,
                3, 3, 1, 2);
  });

  // Copy result back to host and convert
  copy_image(pixels2.get(), dest2.get(), dims2);
  convert_image(pixels, pixels2.get(), dims, dims2);
}

int main(int argc, char** argv) {
  bool do_outputs = false;
  bool do_column_split = false;
  bool do_npp = false;
  bool do_direct = false;
  bool do_gaussian = false;
  bool do_check = false;
  bool do_float2 = false;
  bool do_float = false;
  bool do_half4 = false;
  bool do_half2 = false;

  std::vector<std::string> args;
  for (int i = 1; i < argc; i++) {
    std::string a(argv[i]);
    if (a == "-outputs")
      do_outputs = true;
    else if (a == "-columns")
      do_column_split = true;
    else if (a == "-npp")
      do_npp = true;
    else if (a == "-direct")
      do_direct = true;
    else if (a == "-gaussian")
      do_gaussian = true;
    else if (a == "-check")
      do_check = true;
    else if (a == "-float2")
      do_float2 = true;
    else if (a == "-float")
      do_float = true;
    else if (a == "-half4")
      do_half4 = true;
    else if (a == "-half2")
      do_half2 = true;
    else
      args.push_back(a);
  }

  int radius = (args.size() > 2) ? std::stoi(args[2]) : 5;
  int n_passes = (args.size() > 3) ? std::stoi(args[3]) : 3;

  auto [pixels, _dims] = read_exr(args[0]);
  // Work around stupid "structured binding can't be captured" issue
  auto dims = _dims;

  if (do_check) {
    return run_checks(pixels.get(), dims);
  }

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
      timeit("direct" + std::to_string(outputs), [&]() {
        direct_blur_no_transpose(dest.get(), source.get(), temp.get(), dims,
                                 radius, n_passes, outputs);
      });
    }
  }

  if (do_gaussian) {
    timeit("gaussian", [&]() {
      precomputed_gaussian_blur(dest.get(), source.get(), temp.get(), dims,
                                radius, 2);
    });
  }

  // Current fastest configuration (at 1920x1080, radius 10)
  if (do_float2) {
    convert_and_test<float2>(pixels.get(), dims, radius, n_passes, 2,
                             sizeof(float), "(float2)");
  } else if (do_float) {
    convert_and_test<float>(pixels.get(), dims, radius, n_passes, 1,
                            sizeof(float), "(float)");
  } else if (do_half4) {
    convert_and_test<half4>(pixels.get(), dims, radius, n_passes, 4,
                            sizeof(half), "(half4)");
  } else if (do_half2) {
    convert_and_test<half2>(pixels.get(), dims, radius, n_passes, 2,
                            sizeof(half), "(half2)");
  } else {
    timeit("fastest blur", [&]() {
      smooth_blur(dest.get(), source.get(), temp.get(), dims, radius, n_passes,
                  3, 3, 1, 2);
    });

    // Copy result back to host
    copy_image(pixels.get(), dest.get(), dims);
  }

  write_exr(args[1], dims, pixels.get());

  return 0;
}
