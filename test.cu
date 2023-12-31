#include "cuda_half.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "exr.h"
#include "image.h"
#include "timer.h"

#include "blur.cu"  // Include source directly for now

// ===== Error handling =====
#define cudaCheckError(code)                                           \
  {                                                                    \
    if ((code) != cudaSuccess) {                                       \
      handle_cuda_error(__FILE__, __LINE__, cudaGetErrorString(code)); \
    }                                                                  \
  }

void default_error_handler(const char* file, int line, const char* error) {
  std::cerr << "CUDA failure " << file << ":" << line << " " << error << "\n";
  exit(1);
}

// Called whenever a CUDA function returns an error. Reassign to provide your
// own error handling.
using error_handler_t = void (*)(const char*, int, const char*);
error_handler_t handle_cuda_error = &default_error_handler;

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
      repeated_box_blur(dest.get(), source.get(), temp.get(), dims, radius,
                        n_passes);
      copy_image(baseline.get(), dest.get(), dims);

      // Multiple outputs per thread
      for (int outputs_v = 2; outputs_v <= 3; outputs_v++) {
        for (int outputs_h = 2; outputs_h <= 3; outputs_h++) {
          repeated_box_blur(dest.get(), source.get(), temp.get(), dims, radius,
                            n_passes, outputs_v, outputs_h, 1, 1);
          check_result("outputs " + std::to_string(outputs_v) +
                       std::to_string(outputs_h));

          staggered_blur(dest.get(), source.get(), temp.get(), dims, radius,
                         outputs_h, outputs_v);
          check_result("staggered " + std::to_string(outputs_v) +
                       std::to_string(outputs_h));
        }
      }

      // Column splitting
      for (int threads_v = 2; threads_v <= 4; threads_v++) {
        for (int threads_h = 2; threads_h <= 4; threads_h++) {
          repeated_box_blur(dest.get(), source.get(), temp.get(), dims, radius,
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
void run_benchmark_2(size_t width, size_t height, size_t channel_count,
                     size_t sizeof_channel) {
  size_t stride_bytes =
      (width * sizeof(T) + CUDA_ALIGN - 1) / CUDA_ALIGN * CUDA_ALIGN;

  image_dims dims = {width, height, channel_count, sizeof_channel,
                     stride_bytes / sizeof(T)};

  auto source = cuda_malloc_unique<T>(allocated_bytes(dims));
  auto dest = cuda_malloc_unique<T>(allocated_bytes(dims));
  auto temp = cuda_malloc_unique<T>(allocated_bytes(dims));

  // All black is fine: we don't care about results. We just don't want
  // accidental NaNs or infinities in case those have weird side effects.
  cudaCheckError(cudaMemset(source.get(), 0, allocated_bytes(dims)));

  std::vector<int> radii{1, 3, 5, 10, 15, 20, 25, 30, 50, 75, 100, 200};
  for (auto radius : radii) {
    std::cout << "radius " << radius << "\n";
    std::string best_name;
    float best_time = std::numeric_limits<float>::infinity();

    auto run = [&](auto name, auto callback) {
      float time = timeit(name, callback);
      if (time < best_time) {
        best_name = name;
        best_time = time;
      }
    };

    int outputs_v = 1, outputs_h = 1;
    for (int threads_v = 1; threads_v <= 3; threads_v++) {
      for (int threads_h = 1; threads_h <= 3; threads_h++) {
        run("standard " + std::to_string(outputs_v) +
                std::to_string(outputs_h) + std::to_string(threads_v) +
                std::to_string(threads_h),
            [&]() {
              repeated_box_blur(dest.get(), source.get(), temp.get(), dims,
                                radius, 3, outputs_v, outputs_h, threads_v,
                                threads_h);
            });
      }
    }

    for (int outputs = 1; outputs <= 4; outputs++) {
      run("direct gauss " + std::to_string(outputs), [&]() {
        direct_gaussian_blur(dest.get(), source.get(), temp.get(), dims, radius,
                             outputs);
      });
    }

    if (radius <= MAX_PRECOMPUTED_RADIUS) {
      for (int outputs = 1; outputs <= 4; outputs++) {
        run("precomputed " + std::to_string(outputs), [&]() {
          precomputed_gaussian_blur(dest.get(), source.get(), temp.get(), dims,
                                    radius, outputs);
        });
      }
    }

    for (int outputs = 1; outputs <= 4; outputs++) {
      run("direct " + std::to_string(outputs), [&]() {
        direct_blur_no_transpose(dest.get(), source.get(), temp.get(), dims,
                                 radius, 3, outputs);
      });
    }

    for (int outputs_v = 1; outputs_v <= 4; outputs_v++) {
      for (int outputs_h = 1; outputs_h <= 4; outputs_h++) {
        run("staggered " + std::to_string(outputs_v) +
                std::to_string(outputs_h),
            [&]() {
              staggered_blur(dest.get(), source.get(), temp.get(), dims, radius,
                             outputs_v, outputs_h);
            });
      }
    }

    std::cout << "fastest: " << best_name << " " << best_time << " ms\n\n";
  }
}

static void run_benchmark_1(int width, int height) {
  std::cout << width << "x" << height << "\n";
  std::cout << "float4\n";
  run_benchmark_2<float4>(width, height, 4, sizeof(float));
  std::cout << "\nfloat2\n";
  run_benchmark_2<float2>(width, height, 2, sizeof(float));
  std::cout << "\nfloat\n";
  run_benchmark_2<float>(width, height, 1, sizeof(float));
}

static void run_benchmark() {
  run_benchmark_1(720, 540);
  run_benchmark_1(1920, 1080);
  run_benchmark_1(3840, 2160);
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
    repeated_box_blur(dest2.get(), source2.get(), temp2.get(), dims, radius,
                      n_passes, 3, 3, 1, 2);
  });

  // Copy result back to host and convert
  copy_image(pixels2.get(), dest2.get(), dims2);
  convert_image(pixels, pixels2.get(), dims, dims2);
}

int main(int argc, char** argv) {
  bool do_outputs = false;
  bool do_column_split = false;
  bool do_direct = false;
  bool do_gaussian = false;
  bool do_direct_gaussian = false;
  bool do_staggered = false;
  bool do_check = false;
  bool do_benchmark = false;
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
    else if (a == "-direct")
      do_direct = true;
    else if (a == "-gaussian")
      do_gaussian = true;
    else if (a == "-staggered")
      do_staggered = true;
    else if (a == "-check")
      do_check = true;
    else if (a == "-benchmark")
      do_benchmark = true;
    else if (a == "-float2")
      do_float2 = true;
    else if (a == "-float")
      do_float = true;
    else if (a == "-half4")
      do_half4 = true;
    else if (a == "-half2")
      do_half2 = true;
    else if (a == "-direct-gaussian")
      do_direct_gaussian = true;
    else
      args.push_back(a);
  }

  if (do_benchmark) {
    // No other arguments required
    run_benchmark();
    return 0;
  }

  int radius = (args.size() > 2) ? std::stoi(args[2]) : 5;
  int n_passes = (args.size() > 3) ? std::stoi(args[3]) : 3;

  std::unique_ptr<float4[]> pixels;
  image_dims dims;
  std::tie(pixels, dims) = read_exr(args[0]);

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
                 repeated_box_blur(dest.get(), source.get(), temp.get(), dims,
                                   radius, n_passes, outputs_v, outputs_h, 1,
                                   2);
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
                 repeated_box_blur(dest.get(), source.get(), temp.get(), dims,
                                   radius, n_passes, 1, 1, threads_v,
                                   threads_h);
               });
      }
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

    if (do_direct_gaussian) {
      timeit("direct gaussian", [&]() {
        direct_gaussian_blur(dest.get(), source.get(), temp.get(), dims, radius,
                             2);
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
    } else if (do_staggered) {
      timeit("staggered", [&]() {
        staggered_blur(dest.get(), source.get(), temp.get(), dims, radius, 2,
                       2);
      });

      // Copy result back to host
      copy_image(pixels.get(), dest.get(), dims);
    } else {
      timeit("standard blur", [&]() {
        repeated_box_blur(dest.get(), source.get(), temp.get(), dims, radius,
                          n_passes, 3, 3, 1, 2);
      });

      // Copy result back to host
      copy_image(pixels.get(), dest.get(), dims);
    }

    write_exr(args[1], dims, pixels.get());

    return 0;
  }
}
