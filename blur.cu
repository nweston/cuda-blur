#include <assert.h>
#include <cooperative_groups.h>
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

static int n_blocks(int threads, int block_size, int outputs_per_thread = 1) {
  int block_outputs = block_size * outputs_per_thread;
  return (threads + block_outputs - 1) / block_outputs;
}

// ===== Operators for built-in vector types =====

/// float4 ///
__device__ static float4 operator*(const float4& a, const float b) {
  return {a.x * b, a.y * b, a.z * b, a.w * b};
}

__device__ static float4& operator+=(float4& a, const float4& b) {
  a = {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
  return a;
}

__device__ static float4& operator-=(float4& a, const float4& b) {
  a = {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
  return a;
}

/// float2 ///
__device__ static float2 operator*(const float2& a, const float b) {
  return {a.x * b, a.y * b};
}

__device__ static float2& operator+=(float2& a, const float2& b) {
  a = {a.x + b.x, a.y + b.y};
  return a;
}

__device__ static float2& operator-=(float2& a, const float2& b) {
  a = {a.x - b.x, a.y - b.y};
  return a;
}

// ===== General Utilities =====

// Type of temporary values. This will always be some kind of float, even
// when ImageT is half-float.
template <class ImageT>
struct temp_type {
  using type = ImageT;
};

// Return a single pixel value with all channels set to black.
template <class T>
__device__ static T black(){};

template <>
__device__ float4 black<float4>() {
  return {0, 0, 0, 0};
}
template <>
__device__ float2 black<float2>() {
  return {0, 0};
}
template <>
__device__ float black<float>() {
  return 0;
}

__device__ static int cuda_index_x() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ static int cuda_index_y() {
  return blockIdx.y * blockDim.y + threadIdx.y;
}

// ===== Transpose =====

// Naive transpose
// Supporting vector types with a shared-memory transpose is non-trivial, and
// this is fast enough for now.
template <class ImageT>
__global__ void transpose_kernel(ImageT* dest, const ImageT* source,
                                 image_dims dims) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dims.width || y >= dims.height)
    return;

  dest[x * dims.height + y] = source[pixel_index(dims, x, y)];
}

template <class ImageT>
void transpose(ImageT* dest, const ImageT* source, image_dims dims) {
  dim3 grid(n_blocks(dims.width, 16), n_blocks(dims.height, 16));
  dim3 threads(16, 16);
  transpose_kernel<<<grid, threads>>>(dest, source, dims);
}

// ===== Blur functions =====
template <class ImageT>
__global__ void vertical_box_blur_kernel(ImageT* dest, const ImageT* source,
                                         image_dims dims, int radius) {
  using TempT = typename temp_type<ImageT>::type;

  // Split each column into vertical slices
  int n_slices = blockDim.y;
  int slice_height = (dims.height + n_slices - 1) / n_slices;
  int y_start = threadIdx.y * slice_height;
  int y_limit = min(static_cast<int>(dims.height), y_start + slice_height);

  float scale = 1.0f / (2 * radius + 1);

  for (int x = cuda_index_x(); x < dims.width; x += blockDim.x * gridDim.x) {
    TempT sum;

    // Fill initial box, repeating the edge pixel
    if (y_start == 0) {
      TempT edge = source[pixel_index(dims, x, 0)];
      sum = edge * (radius + 1);
      for (int y = 1; y < radius + 1; y++) {
        sum += source[pixel_index(dims, x, y)];
      }
    } else {
      sum = black<TempT>();
      for (int y = y_start - radius; y < y_start + radius + 1; y++) {
        sum += source[pixel_index(dims, x, max(y, 0))];
      }
    }

    // Compute result pixels
    int top = y_start - radius;
    int bottom = y_start + radius;
    for (int y = y_start; y < y_limit; y++) {
      dest[pixel_index(dims, x, y)] = sum * scale;

      // Shift the box
      sum -= source[pixel_index(dims, x, max(top, 0))];
      top++;
      bottom++;
      sum += source[pixel_index(dims, x, min(bottom, int(dims.height - 1)))];
    }
  }
}

// Vertical box blur, implemented by direct convolution instead of sliding
// window method.
template <class ImageT>
__global__ void vertical_direct_box_blur_kernel(ImageT* dest,
                                                const ImageT* source,
                                                image_dims dims, int radius) {
  using TempT = typename temp_type<ImageT>::type;
  int y = cuda_index_y();
  float scale = 1.0f / (2 * radius + 1);

  for (int x = cuda_index_x(); x < dims.width; x += blockDim.x * gridDim.x) {
    TempT sum = source[pixel_index(dims, x, y)];
    for (int yi = y - radius; yi < y; yi++) {
      sum += source[pixel_index(dims, x, max(yi, 0))];
    }
    for (int yi = y + 1; yi < y + radius + 1; yi++) {
      sum += source[pixel_index(dims, x, min(yi, int(dims.height - 1)))];
    }

    dest[pixel_index(dims, x, y)] = sum * scale;
  }
}

// Horizontal box blur, implemented by direct convolution instead of sliding
// window method.
template <class ImageT>
__global__ void horizontal_direct_box_blur_kernel(ImageT* dest,
                                                  const ImageT* source,
                                                  image_dims dims, int radius) {
  using TempT = typename temp_type<ImageT>::type;
  int y = cuda_index_y();
  float scale = 1.0f / (2 * radius + 1);

  for (int x = cuda_index_x(); x < dims.width; x += blockDim.x * gridDim.x) {
    TempT sum = source[pixel_index(dims, x, y)];
    for (int xi = x - radius; xi < x; xi++) {
      sum += source[pixel_index(dims, max(xi, 0), y)];
    }
    for (int xi = x + 1; xi < x + radius + 1; xi++) {
      sum += source[pixel_index(dims, min(xi, int(dims.width - 1)), y)];
    }

    dest[pixel_index(dims, x, y)] = sum * scale;
  }
}

#include "weights.h"

template <class ImageT>
__global__ void horizontal_precomputed_gaussian_blur_kernel(
    ImageT* dest, const ImageT* source, image_dims dims, int radius) {
  using TempT = typename temp_type<ImageT>::type;
  int y = cuda_index_y();

  for (int x = cuda_index_x(); x < dims.width; x += blockDim.x * gridDim.x) {
    TempT sum = source[pixel_index(dims, x, y)] * weights[radius][0];
    for (int xi = x - radius; xi < x; xi++) {
      int dx = x - xi;
      float w = weights[radius][dx];
      sum += source[pixel_index(dims, max(xi, 0), y)] * w;
    }
    for (int xi = x + 1; xi < x + radius + 1; xi++) {
      int dx = xi - x;
      float w = weights[radius][dx];
      sum += source[pixel_index(dims, min(xi, int(dims.width - 1)), y)] * w;
    }

    dest[pixel_index(dims, x, y)] = sum;
  }
}

template <class ImageT>
__global__ void vertical_precomputed_gaussian_blur_kernel(ImageT* dest,
                                                          const ImageT* source,
                                                          image_dims dims,
                                                          int radius) {
  using TempT = typename temp_type<ImageT>::type;
  int y = cuda_index_y();

  for (int x = cuda_index_x(); x < dims.width; x += blockDim.x * gridDim.x) {
    TempT sum = source[pixel_index(dims, x, y)] * weights[radius][0];
    for (int yi = y - radius; yi < y; yi++) {
      int dy = y - yi;
      float w = weights[radius][dy];
      sum += source[pixel_index(dims, x, max(yi, 0))] * w;
    }
    for (int yi = y + 1; yi < y + radius + 1; yi++) {
      int dy = yi - y;
      float w = weights[radius][dy];
      sum += source[pixel_index(dims, x, min(yi, int(dims.height - 1)))] * w;
    }

    dest[pixel_index(dims, x, y)] = sum;
  }
}

// Blur an image with a repeated box blur.
// 2 passes approximates a triangle filter, 3 is close enough to a Gaussian.
// The radius param indicates the total radius: it will be divided
// more-or-less equally between the passes.
// dest and source may point to the same image (if you don't need to keep the
// source image and want to save some memory).
template <class ImageT>
void smooth_blur(ImageT* dest, const ImageT* source, ImageT* temp,
                 image_dims dims, int radius, int n_passes,
                 int outputs_per_thread_v = 1, int outputs_per_thread_h = 1,
                 int threads_per_column_v = 1, int threads_per_column_h = 1) {
  const int BLOCK_WIDTH = 128;

  // Each blur pass and transpose needs to read from one image and write to
  // another. To avoid any copying, we swap these pointers around after each
  // operation.
  ImageT* from = dest;
  ImageT* to = temp;

  // Ensure that each pass will have radius >= 1. This speeds up small blurs
  // by reducing the number of passes.
  n_passes = std::min(n_passes, radius);

  // Vertical blur
  {
    int remaining = radius;
    dim3 block_dim(BLOCK_WIDTH, threads_per_column_v);
    int grid_dim = n_blocks(dims.width, BLOCK_WIDTH, outputs_per_thread_v);
    for (int i = 0; i < n_passes; i++) {
      int this_radius = remaining / (n_passes - i);
      remaining -= this_radius;
      // On the very first pass, read from the source image
      const ImageT* s = (i == 0) ? source : from;
      vertical_box_blur_kernel<<<grid_dim, block_dim>>>(to, s, dims,
                                                        this_radius);
      std::swap(to, from);
    }
  }

  transpose(to, from, dims);
  std::swap(to, from);

  // Horizontal blur
  {
    int remaining = radius;
    // Transpose turns any horizontal padding into vertical padding. Ignore
    // those extra pixels when blurring.
    image_dims transpose_dims = {dims.height, dims.width, dims.channel_count,
                                 dims.sizeof_channel, dims.height};
    dim3 block_dim(BLOCK_WIDTH, threads_per_column_h);
    int grid_dim =
        n_blocks(transpose_dims.width, BLOCK_WIDTH, outputs_per_thread_h);
    for (int i = 0; i < n_passes; i++) {
      int this_radius = remaining / (n_passes - i);
      remaining -= this_radius;
      vertical_box_blur_kernel<<<grid_dim, block_dim>>>(
          to, from, transpose_dims, this_radius);
      std::swap(to, from);
    }
  }

  // Transpose back to the original format. This version of the dims includes
  // the extra height.
  transpose(to, from,
            {dims.height, dims.stride_pixels, dims.channel_count,
             dims.sizeof_channel, dims.height});
  assert(to == dest);
}

// Use horizontal and vertical direct blur kernels, without transposing.
template <class ImageT>
void direct_blur_no_transpose(ImageT* dest, const ImageT* source, ImageT* temp,
                              image_dims dims, int radius, int n_passes,
                              int outputs_per_thread = 1) {
  dim3 BLOCK_DIM(16, 8);

  // Each blur pass and transpose needs to read from one image and write to
  // another. To avoid any copying, we swap these pointers around after each
  // operation.
  ImageT* from = dest;
  ImageT* to = temp;

  // Ensure that each pass will have radius >= 1. This speeds up small blurs
  // by reducing the number of passes.
  n_passes = std::min(n_passes, radius);

  dim3 grid_dim(n_blocks(dims.width, BLOCK_DIM.x, outputs_per_thread),
                n_blocks(dims.height, BLOCK_DIM.y));

  // Horizontal blur
  {
    int remaining = radius;
    for (int i = 0; i < n_passes; i++) {
      int this_radius = remaining / (n_passes - i);
      remaining -= this_radius;
      // On the very first pass, read from the source image
      const ImageT* s = (i == 0) ? source : from;
      horizontal_direct_box_blur_kernel<<<grid_dim, BLOCK_DIM>>>(to, s, dims,
                                                                 this_radius);
      std::swap(to, from);
    }
  }

  // Vertical blur
  {
    int remaining = radius;

    for (int i = 0; i < n_passes; i++) {
      int this_radius = remaining / (n_passes - i);
      remaining -= this_radius;
      vertical_direct_box_blur_kernel<<<grid_dim, BLOCK_DIM>>>(to, from, dims,
                                                               this_radius);
      std::swap(to, from);
    }
  }

  assert(from == dest);
}

template <class ImageT>
void precomputed_gaussian_blur(ImageT* dest, const ImageT* source, ImageT* temp,
                               image_dims dims, int radius,
                               int outputs_per_thread = 1) {
  dim3 BLOCK_DIM(16, 8);
  dim3 grid_dim(n_blocks(dims.width, BLOCK_DIM.x, outputs_per_thread),
                n_blocks(dims.height, BLOCK_DIM.y));

  assert(radius <= MAX_PRECOMPUTED_RADIUS);

  horizontal_precomputed_gaussian_blur_kernel<<<grid_dim, BLOCK_DIM>>>(
      temp, source, dims, radius);
  vertical_precomputed_gaussian_blur_kernel<<<grid_dim, BLOCK_DIM>>>(
      dest, temp, dims, radius);
}
