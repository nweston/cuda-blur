#include "exr.h"
#include <OpenEXR/ImfRgbaFile.h>
#include <assert.h>
#include "OpenEXR/ImfRgba.h"

const int CUDA_ALIGN = 256;

// Convert RGBA image from half to float. Add padding appropriate for CUDA
// (but still return an image in host memory).
static std::pair<std::unique_ptr<float4[]>, image_dims> half_image_to_float(
    const Imf::Rgba *pixels, size_t width, size_t height) {
  // Round stride up to a multiple of the alignment size
  size_t stride_bytes =
      (width * sizeof(float4) + CUDA_ALIGN - 1) / CUDA_ALIGN * CUDA_ALIGN;

  image_dims dims = {width, height, 4, sizeof(float),
                     stride_bytes / sizeof(float4)};

  auto float_pixels = std::make_unique<float4[]>(allocated_pixels(dims));
  for (size_t y = 0; y < dims.height; y++) {
    for (size_t x = 0; x < dims.width; x++) {
      auto p = pixels[y * width + x];
      float_pixels.get()[pixel_index(dims, x, y)] = {p.r, p.g, p.b, p.a};
    }
  }

  return {std::move(float_pixels), dims};
}

// Convert RGBA image from float to half. Remove padding.
static std::unique_ptr<Imf::Rgba[]> float_image_to_half(const float4 *pixels,
                                                        image_dims dims) {
  auto half_pixels = std::make_unique<Imf::Rgba[]>(dims.width * dims.height);
  for (size_t y = 0; y < dims.height; y++) {
    for (size_t x = 0; x < dims.width; x++) {
      auto p = pixels[pixel_index(dims, x, y)];
      half_pixels.get()[y * dims.width + x] = {p.x, p.y, p.z, p.w};
    }
  }

  return std::move(half_pixels);
}

std::pair<std::unique_ptr<float4[]>, image_dims> read_exr(
    const std::string &filename) {
  Imf::RgbaInputFile file(filename.c_str());
  auto dw = file.dataWindow();
  size_t width = dw.max.x - dw.min.x + 1;
  size_t height = dw.max.y - dw.min.y + 1;

  auto pixels = std::make_unique<Imf::Rgba[]>(width * height);
  file.setFrameBuffer(
      reinterpret_cast<Imf::Rgba *>(pixels.get()) - dw.min.x - dw.min.y * width,
      1, width);
  file.readPixels(dw.min.y, dw.max.y);

  return half_image_to_float(pixels.get(), width, height);
}

void write_exr(const std::string &filename, image_dims dims,
               const float4 *pixels) {
  assert(dims.channel_count == 4);

  auto half_pixels = float_image_to_half(pixels, dims);

  Imf::RgbaOutputFile file(filename.c_str(), dims.width, dims.height,
                           Imf::WRITE_RGBA);
  file.setFrameBuffer(half_pixels.get(), 1, dims.width);
  file.writePixels(dims.height);
}
