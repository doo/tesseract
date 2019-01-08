///////////////////////////////////////////////////////////////////////
// File:        intsimdmatrix.cpp
// Description: Base class for 8-bit int SIMD matrix multipliers.
// Author:      Ray Smith
// Created:     Tue Aug 15 08:01:32 PST 2017
//
// (C) Copyright 2017, Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
///////////////////////////////////////////////////////////////////////

#include "intsimdmatrix.h"
#include "genericvector.h"      // for GenericVector
#include "intsimdmatrixavx2.h"  // for IntSimdMatrixAVX2
#include "intsimdmatrixsse.h"   // for IntSimdMatrixSSE
#include "matrix.h"             // for GENERIC_2D_ARRAY
#include "simddetect.h"         // for SIMDDetect

namespace tesseract {

// Factory makes and returns an IntSimdMatrix (sub)class of the best
// available type for the current architecture.
/* static */
IntSimdMatrix* IntSimdMatrix::GetFastestMultiplier() {
  IntSimdMatrix* multiplier = nullptr;
  if (SIMDDetect::IsAVX2Available()) {
    multiplier = new IntSimdMatrixAVX2();
  } else if (SIMDDetect::IsSSEAvailable()) {
    multiplier = new IntSimdMatrixSSE();
  } else {
    // Default c++ implementation.
    multiplier = new IntSimdMatrix();
  }
  return multiplier;
}

// Computes a reshaped copy of the weight matrix w. If there are no
// partial_funcs_, it does nothing.
void IntSimdMatrix::Init(const GENERIC_2D_ARRAY<int8_t>& w) {
  if (partial_funcs_.empty()) return;
  int num_out = w.dim1();
  int num_in = w.dim2() - 1;
  // The rounded-up sizes of the reshaped weight matrix, excluding biases.
  int rounded_num_in = Roundup(num_in, num_inputs_per_group_);
  int rounded_num_out = RoundOutputs(num_out);
  // Add the bias and compute the required size.
  shaped_w_.resize((rounded_num_in + 1) * rounded_num_out, 0);
  int shaped_index = 0;
  int output = 0;
  // Each number of registers needs a different format! Iterates over the
  // different numbers of registers (each a power of 2).
  for (int num_registers = max_output_registers_; num_registers >= 1;
       num_registers /= 2) {
    // The number of outputs that we will generate with this many registers.
    int num_outputs_per_register_set =
        num_registers * num_outputs_per_register_;
    // Use the max number of registers until we have to go fewer.
    while (output + num_outputs_per_register_set <= rounded_num_out) {
      // Accumulating outputs in registers saves iterating over the inputs, so
      // we only have to do it once per output register set.
      for (int input = 0; input < num_in; input += num_inputs_per_group_) {
        // Iterate over the number of outputs in a register set.
        for (int j = 0; j < num_outputs_per_register_set; ++j) {
          // Inner-most loop corresponds to the number of inputs in an input
          // group.
          for (int i = 0; i < num_inputs_per_group_; ++i) {
            int8_t weight = 0;
            if (output + j < num_out && input + i < num_in)
              weight = w(output + j, input + i);
            shaped_w_[shaped_index++] = weight;
          }
        }
      }
      // Append the bias weights for the register set.
      for (int j = 0; j < num_outputs_per_register_set; ++j) {
        int8_t weight = 0;
        if (output + j < num_out) weight = w(output + j, num_in);
        shaped_w_[shaped_index++] = weight;
      }
      output += num_outputs_per_register_set;
    }
  }
}
    
#ifdef __ARM_NEON__

#include <arm_neon.h>
    
    
#define SIMD_NEON_PREFECH_SIZE 384

#define SIMD_VEC_SET1_EPI8(a) \
{a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a}

#define SIMD_VEC_SET1_EPI16(a) \
{a, a, a, a, a, a, a, a}
    
#define SIMD_VEC_SET1_EPI32(a) \
{a, a, a, a}
    
#define SIMD_VEC_SET1_EPI64(a) \
{a, a}

#define SIMD_VEC_SET1_PI16(a) \
{SIMD_LL_SET1_EPI16(a)}

    const size_t A = sizeof(uint8x16_t);
    
    const uint8x16_t K8_00 = SIMD_VEC_SET1_EPI8(0x00);
    const uint8x16_t K8_FF = SIMD_VEC_SET1_EPI8(0xFF);
    const uint32x4_t K32_00000000 = SIMD_VEC_SET1_EPI32(0x00000000);
    const uint64x2_t K64_0000000000000000 = SIMD_VEC_SET1_EPI64(0x0000000000000000);
    
    size_t AlignLo(size_t size, size_t align)
    {
        return size & ~(align - 1);
    }
    
    void * AlignLo(const void * ptr, size_t align)
    {
        return (void *)(((size_t)ptr) & ~(align - 1));
    }
    
    bool Aligned(const void * ptr, size_t align = sizeof(uint8x16_t))
    {
        return ptr == AlignLo(ptr, align);
    }
    
    template <bool align> int8x16_t Load(const int8_t * p);
    
    template <> int8x16_t Load<false>(const int8_t * p)
    {
#if defined(__GNUC__) && SIMD_NEON_PREFECH_SIZE
        __builtin_prefetch(p + SIMD_NEON_PREFECH_SIZE);
#endif
        return vld1q_s8(p);
    }
    
    template <> int8x16_t Load<true>(const int8_t * p)
    {
#if defined(__GNUC__)
#if SIMD_NEON_PREFECH_SIZE
        __builtin_prefetch(p + SIMD_NEON_PREFECH_SIZE);
#endif
        int8_t * _p = (int8_t *)__builtin_assume_aligned(p, 16);
        return vld1q_s8(_p);
#elif defined(_MSC_VER)
        return vld1q_s8_ex(p, 128);
#else
        return vld1q_s8(p);
#endif
    }
    
    uint8x16_t ShiftLeft(uint8x16_t value, size_t shift)
    {
        if (shift & 8)
            value = vextq_u8(K8_00, value, 8);
        if (shift & 4)
            value = vextq_u8(K8_00, value, 12);
        if (shift & 2)
            value = vextq_u8(K8_00, value, 14);
        if (shift & 1)
            value = vextq_u8(K8_00, value, 15);
        return value;
    }
    
    int32_t ExtractSum32i(const int32x4_t & a)
    {
        return vgetq_lane_s32(a, 0) + vgetq_lane_s32(a, 1) + vgetq_lane_s32(a, 2) + vgetq_lane_s32(a, 3);
    }

    template <int part> int8x8_t Half(int8x16_t a);
    
    template <> int8x8_t Half<0>(int8x16_t a)
    {
        return vget_low_s8(a);
    }
    
    template <> int8x8_t Half<1>(int8x16_t a)
    {
        return vget_high_s8(a);
    }

    int32x4_t Correlation(const int8x16_t & a, const int8x16_t & b)
    {
        int16x8_t lo = vmull_s8(Half<0>(a), Half<0>(b));
        int16x8_t hi = vmull_s8(Half<1>(a), Half<1>(b));
        return vaddq_s32(vpaddlq_s16(lo), vpaddlq_s16(hi));
    }
    
    template <bool align> void CorrelationSum(const int8_t * a, const int8_t * b, size_t width, int32_t * sum)
    {
        assert(width >= A);
        if (align)
            assert(Aligned(a) && Aligned(b));
        
        size_t alignedWidth = AlignLo(width, A);
        uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
        
        int32x4_t rowSum = K32_00000000;
        for (size_t col = 0; col < alignedWidth; col += A)
        {
            int8x16_t _a = Load<align>(a + col);
            int8x16_t _b = Load<align>(b + col);
            rowSum = vaddq_s32(rowSum, Correlation(_a, _b));
        }
        if (alignedWidth != width)
        {
            int8x16_t _a = vandq_s8(Load<false>(a + width - A), tailMask);
            int8x16_t _b = vandq_s8(Load<false>(b + width - A), tailMask);
            rowSum = vaddq_s32(rowSum, Correlation(_a, _b));
        }

        *sum = ExtractSum32i(rowSum);
    }
    
    void CorrelationSum(const int8_t * a, const int8_t * b, size_t width, int32_t * sum)
    {
        if (width < A) {
            *sum = 0;
            for (int i = 0; i < width; ++i) {
                *sum += a[i] * b[i];
            }
            return;
        }

        if (Aligned(a) && Aligned(b))
            CorrelationSum<true>(a, b, width, sum);
        else
            CorrelationSum<false>(a, b, width, sum);
    }
    
#else
    void CorrelationSum(const int8_t * a, const int8_t * b, size_t width, int32_t * sum)
    {
        *sum = 0;
        for (int i = 0; i < width; ++i) {
            *sum += a[i] * b[i];
        }
    }
#endif

// Computes matrix.vector v = Wu.
// u is of size W.dim2() - 1 and the output v is of size W.dim1().
// u is imagined to have an extra element at the end with value 1, to
// implement the bias, but it doesn't actually have it.
void IntSimdMatrix::MatrixDotVector(const GENERIC_2D_ARRAY<int8_t>& w,
                                    const GenericVector<double>& scales,
                                    const int8_t* u, double* v) const {
  int num_out = w.dim1();
  int num_in = w.dim2() - 1;
  if (partial_funcs_.empty()) {
    // Base implementation.
    for (int i = 0; i < num_out; ++i) {
      const int8_t* wi = w[i];
      int total = 0;
        CorrelationSum(wi, u, num_in, &total);
//      for (int j = 0; j < num_in; ++j) total += wi[j] * u[j];
      // Add in the bias and correct for integer values.
      v[i] = (static_cast<double>(total) / INT8_MAX + wi[num_in]) * scales[i];
    }
  } else {
    const int8_t* w_data = shaped_w_.data();
    const double* scales_data = &scales[0];
    // Each call to a partial_func_ produces group_size outputs, except the
    // last one, which can produce less.
    int group_size = num_outputs_per_register_ * max_output_registers_;
    int rounded_num_in = Roundup(num_in, num_inputs_per_group_);
    int rounded_num_out = RoundOutputs(num_out);
    int output = 0;
    for (auto fn : partial_funcs_) {
      // The amount of w_data consumed by each call to fn.
      int w_step = (rounded_num_in + 1) * group_size;
      // Run with this group size, until it would produce too much output, then
      // switch to a smaller size.
      for (; output + group_size <= rounded_num_out; output += group_size) {
        (*fn)(w_data, scales_data, u, rounded_num_in, num_out - output, v);
        w_data += w_step;
        scales_data += group_size;
        v += group_size;
      }
      group_size /= 2;
    }
  }
}

}  // namespace tesseract
