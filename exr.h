//  -*- mode: c++ -*-
#ifndef __EXR_H
#define __EXR_H

#include <cuda_runtime_api.h>
#include <memory>
#include <utility>
#include "image.h"

// Read an EXR file, convert to float, and add padding.
std::pair<std::unique_ptr<float4[]>, image_dims> read_exr(
    const std::string &filename);
// Convert image to half and write to an EXR file.
void write_exr(const std::string &filename, image_dims dims,
               const float4 *pixels);

#endif  // __EXR_H
