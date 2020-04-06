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