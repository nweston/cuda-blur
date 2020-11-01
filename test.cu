#include "cuda_half.h"

#include <memory>
#include <string>
#include <utility>
#include "exr.h"
#include "image.h"

#include "blur.cu"  // Include source directly for now

int main(int argc, char **argv) {
  auto [pixels, dims] = read_exr(argv[1]);
  write_exr(argv[2], dims, pixels.get());
  return 0;
}
