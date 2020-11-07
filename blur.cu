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
__device__ static int cuda_index() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void vertical_box_blur_kernel(ImageT* dest, const ImageT* source,
                                         image_dims dims, int radius) {
  int x = cuda_index();
  if (x >= dims.width)
    return;

  float scale = 1.0f / (2 * radius + 1);

  // Fill initial box, repeating the edge pixel
  TempT edge = source[pixel_index(dims, x, 0)];
  TempT sum = edge * (radius + 1);
  for (int y = 1; y < radius + 1; y++) {
    sum += source[pixel_index(dims, x, y)];
  }

  // Compute result pixels
  int top = -radius;
  int bottom = radius;
  for (int y = 0; y < dims.height; y++) {
    dest[pixel_index(dims, x, y)] = sum * scale;

    // Shift the box
    sum -= source[pixel_index(dims, x, max(top, 0))];
    top++;
    bottom++;
    sum += source[pixel_index(dims, x, min(bottom, int(dims.height - 1)))];
  }
}

void vertical_box_blur(ImageT* dest, const ImageT* source, image_dims dims,
                       int radius) {
  const int BLOCK_SIZE = 128;
  int grid_dim = n_blocks(dims.width, BLOCK_SIZE);
  vertical_box_blur_kernel<<<grid_dim, BLOCK_SIZE>>>(dest, source, dims,
                                                     radius);
}
