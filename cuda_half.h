//  -*- mode: c++ -*-
#ifndef __CUDA_HALF_H
#define __CUDA_HALF_H

#ifdef __CUDACC__
// In CUDA code, we need the normal definition of the half type.
#include <cuda_fp16.h>
using cuda_half = half;
#else
// In C++ code, avoid conflict between CUDA and OpenEXR half types.
#define CUDA_NO_HALF
#define __CUDA_NO_HALF_CONVERSIONS__
#include <cuda_fp16.h>
using cuda_half = __half;
#endif

#endif  // __CUDA_HALF_H
