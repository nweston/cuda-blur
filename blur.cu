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

static int n_blocks(int threads, int block_size) {
  return (threads + block_size - 1) / block_size;
}

// ===== Operators for built-in vector types =====
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

// ===== General Utilities =====

// Type used for image data. This will eventually be templated to support
// different vector lengths, as well as half-float data.
using ImageT = float4;
// Type of temporary values. This will always be some kind of float, even
// when ImageT is half-float.
using TempT = float4;

__device__ static int cuda_index() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

static cudaTextureObject_t create_texture(const ImageT* source,
                                          image_dims dims) {
  cudaTextureDesc tex_desc = {};
  tex_desc.addressMode[0] = cudaAddressModeClamp;
  tex_desc.addressMode[1] = cudaAddressModeClamp;
  tex_desc.filterMode = cudaFilterModePoint;
  tex_desc.readMode = cudaReadModeElementType;
  tex_desc.normalizedCoords = 0;

  cudaResourceDesc resource_desc = {cudaResourceTypePitch2D};
  resource_desc.res.pitch2D.devPtr = const_cast<ImageT*>(source);
  resource_desc.res.pitch2D.desc = cudaCreateChannelDesc<ImageT>();
  resource_desc.res.pitch2D.width = dims.width;
  resource_desc.res.pitch2D.height = dims.height;
  resource_desc.res.pitch2D.pitchInBytes = stride_bytes(dims);

  cudaTextureObject_t tex = 0;
  cudaCheckError(
      cudaCreateTextureObject(&tex, &resource_desc, &tex_desc, nullptr));
  return tex;
}

// ===== Transpose =====

// Naive transpose
// Supporting vector types with a shared-memory transpose is non-trivial, and
// this is fast enough for now.
__global__ void transpose_kernel(ImageT* dest, const ImageT* source,
                                 image_dims dims) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dims.width || y >= dims.height)
    return;

  dest[x * dims.height + y] = source[pixel_index(dims, x, y)];
}

void transpose(ImageT* dest, const ImageT* source, image_dims dims) {
  dim3 grid(n_blocks(dims.width, 16), n_blocks(dims.height, 16));
  dim3 threads(16, 16);
  transpose_kernel<<<grid, threads>>>(dest, source, dims);
}

// ===== Blur functions =====
__global__ void vertical_box_blur_kernel(ImageT* dest, const ImageT* source,
                                         image_dims dims, int radius) {
  // Split each column into vertical slices
  int n_slices = blockDim.y;
  int slice_height = (dims.height + n_slices - 1) / n_slices;
  int y_start = threadIdx.y * slice_height;
  int y_limit = min(static_cast<int>(dims.height), y_start + slice_height);

  float scale = 1.0f / (2 * radius + 1);

  for (int x = cuda_index(); x < dims.width; x += blockDim.x * gridDim.x) {
    TempT sum;

    // Fill initial box, repeating the edge pixel
    if (y_start == 0) {
      TempT edge = source[pixel_index(dims, x, 0)];
      sum = edge * (radius + 1);
      for (int y = 1; y < radius + 1; y++) {
        sum += source[pixel_index(dims, x, y)];
      }
    } else {
      sum = {0, 0, 0, 0};
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

// Horizontal box blur using texture. This is significantly slower (approx
// 2x) than transposing and using the vertical blur.
__global__ void horizontal_box_blur_kernel(ImageT* dest,
                                           cudaTextureObject_t source,
                                           image_dims dims, int radius) {
  int y = cuda_index();
  if (y >= dims.height)
    return;

  float scale = 1.0f / (2 * radius + 1);

  // Fill initial box, repeating the edge pixel
  TempT edge = tex2D<ImageT>(source, 0, y);
  TempT sum = edge * (radius + 1);
  for (int x = 1; x < radius + 1; x++) {
    sum += tex2D<ImageT>(source, x, y);
  }

  // Compute result pixels
  int left = -radius;
  int right = radius;
  for (int x = 0; x < dims.width; x++) {
    dest[pixel_index(dims, x, y)] = sum * scale;

    // Shift the box
    sum -= tex2D<ImageT>(source, left, y);
    left++;
    right++;
    sum += tex2D<ImageT>(source, right, y);
  }
}

void vertical_box_blur(ImageT* dest, const ImageT* source, image_dims dims,
                       int radius) {
  const int BLOCK_SIZE = 128;
  int grid_dim = n_blocks(dims.width, BLOCK_SIZE);
  vertical_box_blur_kernel<<<grid_dim, BLOCK_SIZE>>>(dest, source, dims,
                                                     radius);
}

void box_blur(ImageT* dest, const ImageT* source, ImageT* temp, image_dims dims,
              int radius) {
  const int BLOCK_SIZE = 128;

  // Vertical blur
  {
    int grid_dim = n_blocks(dims.width, BLOCK_SIZE);
    vertical_box_blur_kernel<<<grid_dim, BLOCK_SIZE>>>(temp, source, dims,
                                                       radius);
  }

  transpose(dest, temp, dims);
  image_dims transpose_dims = {dims.height, dims.width, dims.channel_count,
                               dims.sizeof_channel, dims.height};

  // Horizontal blur
  {
    // Transpose turns any horizontal padding into vertical padding. Ignore
    // those extra pixels when blurring.
    int grid_dim = n_blocks(transpose_dims.width, BLOCK_SIZE);
    vertical_box_blur_kernel<<<grid_dim, BLOCK_SIZE>>>(temp, dest,
                                                       transpose_dims, radius);
  }

  // Transpose back to the original format.
  transpose(dest, temp, transpose_dims);
}

// Blur an image with a repeated box blur.
// 2 passes approximates a triangle filter, 3 is close enough to a Gaussian.
// The radius param indicates the total radius: it will be divided
// more-or-less equally between the passes.
// dest and source may point to the same image (if you don't need to keep the
// source image and want to save some memory).
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

  // Vertical blur
  {
    int remaining = radius;
    dim3 block_dim(BLOCK_WIDTH, threads_per_column_v);
    int grid_dim = n_blocks(dims.width, BLOCK_WIDTH) / outputs_per_thread_v;
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
        n_blocks(transpose_dims.width, BLOCK_WIDTH) / outputs_per_thread_h;
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

// Like smooth_blur, but use textures for horizontal blur
void smooth_blur_texture(ImageT* dest, const ImageT* source, ImageT* temp,
                         image_dims dims, int radius, int n_passes,
                         int outputs_per_thread_v = 1,
                         int outputs_per_thread_h = 1) {
  const int BLOCK_SIZE = 128;

  // Each blur pass and transpose needs to read from one image and write to
  // another. To avoid any copying, we swap these pointers around after each
  // operation.
  ImageT* from = dest;
  ImageT* to = temp;

  // Vertical blur
  {
    int remaining = radius;
    int grid_dim = n_blocks(dims.width, BLOCK_SIZE) / outputs_per_thread_v;
    for (int i = 0; i < n_passes; i++) {
      int this_radius = remaining / (n_passes - i);
      remaining -= this_radius;
      // On the very first pass, read from the source image
      const ImageT* s = (i == 0) ? source : from;
      vertical_box_blur_kernel<<<grid_dim, BLOCK_SIZE>>>(to, s, dims,
                                                         this_radius);
      std::swap(to, from);
    }
  }

  // Horizontal blur
  {
    int remaining = radius;
    int grid_dim = n_blocks(dims.height, BLOCK_SIZE) / outputs_per_thread_h;
    for (int i = 0; i < n_passes; i++) {
      int this_radius = remaining / (n_passes - i);
      remaining -= this_radius;

      if (false) {
        cudaMemcpy(to, from, allocated_bytes(dims), cudaMemcpyDefault);
      } else {
        auto tex = create_texture(from, dims);
        horizontal_box_blur_kernel<<<grid_dim, BLOCK_SIZE>>>(to, tex, dims,
                                                             this_radius);
        cudaCheckError(cudaDestroyTextureObject(tex));
      }

      std::swap(to, from);
    }
  }
}
