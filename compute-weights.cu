// Pre-compute weights for direct convolution blurs.

#include <cstring>
#include <iostream>
#include <memory>

#include "blur.cu"

__global__ void make_diagonal_image(float *image, image_dims dims) {
  int x = cuda_index_x();
  int y = cuda_index_y();

  if (x >= dims.width || y >= dims.height)
    return;

  image[pixel_index(dims, x, y)] = (x == y) ? 1.0f : 0.0f;
}

void vertical_blur(float *output, float *temp, const float *input,
                   image_dims dims, int radius) {
  int remaining = radius;
  int r = radius / 3;
  vertical_box_blur_kernel<<<1, 128>>>(output, input, dims, r);

  remaining -= r;
  r = remaining / 2;
  vertical_box_blur_kernel<<<1, 128>>>(temp, output, dims, r);

  r = remaining - r;
  vertical_box_blur_kernel<<<1, 128>>>(output, temp, dims, r);
}

int main() {
  const int MAX_RADIUS = 30;
  std::cout << "////// Pre-computed weights for Gaussian blur /////\n"
            << "// Generated by compute-weights.cxx.\n\n"
            << "const int MAX_PRECOMPUTED_RADIUS = " << MAX_RADIUS << ";\n";

  // Make big images so we don't have to worry about going off the end.
  size_t size = 3 * MAX_RADIUS * MAX_RADIUS * sizeof(float);
  float *input;
  float *temp;
  float *output;
  cudaMalloc(&input, size);
  cudaMalloc(&temp, size);
  cudaMalloc(&output, size);

  for (int radius = 1; radius <= MAX_RADIUS; radius++) {
    // Create a diagonal image (with 1.0 where x==y, and 0.0 elsewhere), and
    // blur it vertically in order to characterize the response of the
    // iterated box filter. This is different at the edges (see comment in
    // blur.cu), so we need this whole image to cover all the cases.
    image_dims dims{size_t(radius + 1), size_t(2 * radius + 1), 1,
                    sizeof(float), size_t(radius + 1)};
    make_diagonal_image<<<dim3(n_blocks(dims.width, 16),
                               n_blocks(dims.height, 16)),
                          dim3(16, 16)>>>(input, dims);
    vertical_blur(output, temp, input, dims, radius);

    auto weights = std::make_unique<float[]>(allocated_pixels(dims));
    cudaCheckError(cudaMemcpy(weights.get(), output, allocated_bytes(dims),
                              cudaMemcpyDefault));

    // In the last column, the impulse is centered in the image and the blur
    // radius doesn't touch either edge, so the values there give the normal
    // weights.
    std::cout << "__constant__ float weights" << std::to_string(radius)
              << "[] = {";
    for (int i = radius; i <= 2 * radius; i++) {
      std::cout << weights.get()[pixel_index(dims, radius, i)] << ", ";
    }
    std::cout << "};\n";

    // Now find special-case weights for the edges
    // Reading across column N gives the weights for an output pixel which
    // is N from the edge.
    for (int center = 0; center < radius; center++) {
      std::cout << "__constant__ float edge_weights" << radius << "_" << center
                << "[] = {";

      for (int i = 0; i < radius; i++) {
        std::cout << weights.get()[pixel_index(dims, i, center)] << ", ";
      }
      std::cout << "};\n";
    }
    std::cout << "__constant__ float* edge_weights" << radius << "[] = {\n";
    for (int i = 0; i < radius; i++) {
      std::cout << "edge_weights" << radius << "_" << i << ",\n";
    }
    std::cout << "};\n";
  }
  std::cout << "__constant__ float* weights[] = {nullptr, ";
  for (int radius = 1; radius <= MAX_RADIUS; radius++) {
    std::cout << "weights" << radius << ", ";
  }
  std::cout << "};\n";
  std::cout << "__constant__ float** edge_weights[] = {nullptr, ";
  for (int radius = 1; radius <= MAX_RADIUS; radius++) {
    std::cout << "edge_weights" << radius << ", ";
  }
  std::cout << "};\n";

  std::cout << "////// End pre-computed weights /////\n";

  return 0;
}
