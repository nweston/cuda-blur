//  -*- mode: c++ -*-

#ifndef __IMAGE_H
#define __IMAGE_H

namespace cuda_blur {

// Describes the dimensions and memory layout of an image.
struct image_dims {
  size_t width;
  size_t height;
  size_t channel_count;
  size_t sizeof_channel;  // in bytes
  size_t stride_pixels;
};

__host__ __device__ inline size_t pixel_size(const image_dims& dims) {
  return dims.channel_count * dims.sizeof_channel;
}

// Offset in bytes from one line to the next
__host__ __device__ inline size_t stride_bytes(const image_dims& dims) {
  return dims.stride_pixels * pixel_size(dims);
}

// Total bytes needed to store the image
__host__ __device__ inline size_t allocated_bytes(const image_dims& dims) {
  return stride_bytes(dims) * dims.height;
}

// Total pixels needed to store the image
__host__ __device__ inline size_t allocated_pixels(const image_dims& dims) {
  return dims.stride_pixels * dims.height;
}

// Array index for accessing the given pixel
__host__ __device__ inline size_t pixel_index(const image_dims& dims, int x,
                                              int y) {
  return y * dims.stride_pixels + x;
}

}  // namespace cuda_blur
#endif  // __IMAGE_H
