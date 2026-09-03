/*
SHADER.METAL

L0 shaders
*/


#include <metal_stdlib>
using namespace metal;


// common definitions parallel to common.hpp

using uint = uint32_t;
using iint = int32_t;

typedef int32_t int32;
typedef float float32;

#define MAXDIMS 32
#define MAXSEEDF 4294967295.0f


// functions

#define binary_op(dtype, name, op) \
kernel void name( \
    const device dtype* a [[buffer(0)]], \
    const device dtype* b [[buffer(1)]], \
    device dtype* out [[buffer(2)]], \
    constant uint& ndim [[buffer(3)]], \
    constant int* a_strides [[buffer(4)]], \
    constant uint& a_offset [[buffer(5)]], \
    constant int* b_strides [[buffer(6)]], \
    constant uint& b_offset [[buffer(7)]], \
    constant int* out_strides [[buffer(8)]], \
    uint id [[thread_position_in_grid]] \
) { \
    uint out_idx = id; \
    uint a_idx = a_offset; \
    uint b_idx = b_offset; \
    uint coord; \
\
    /* broadcasting ensures that for every indices I = [i_1, i_2, ...] of out, a[I] + b[I] = out[I] */ \
\
    for (uint i = 0; i < ndim; i++) { \
        /* generate indices of out view from idx of out buffer (i.e. thread id) */ \
        coord = out_idx / out_strides[i]; \
        out_idx %= out_strides[i]; \
        /* use out view idx (coord) to generate a&b buffer indices */ \
        a_idx += coord * a_strides[i]; \
        b_idx += coord * b_strides[i]; \
    } \
\
    out[id] = a[a_idx] op b[b_idx]; \
}

#define unary_op(dtype, name, op) \
kernel void name( \
    device dtype* a [[buffer(0)]], \
    const device dtype* b [[buffer(1)]], \
    constant uint& ndim [[buffer(2)]], \
    constant int* a_strides [[buffer(3)]], \
    constant uint& a_offset [[buffer(4)]], \
    constant int* b_strides [[buffer(5)]], \
    constant uint& b_offset [[buffer(6)]], \
    uint id [[thread_position_in_grid]] \
) { \
    uint a_idx = id; \
    uint b_idx = b_offset; \
    uint coord; \
\
    /* broadcasting ensures that for every indices I = [i_1, i_2, ...] of a, a[I] = a[I] + b[I] */ \
\
    for (uint i = 0; i < ndim; i++) { \
        /* convert buffer index (relative to offset) to view indices */ \
        coord = a_idx / a_strides[i]; \
        a_idx %= a_strides[i]; \
        /* view indices to buffer index of b */ \
        b_idx += coord * b_strides[i]; \
    } \
\
    a[id + a_offset] op b[b_idx]; \
}


// NOT THE FINAL PRNG IMPLEMENTATION.
// xorshift is being used as a placeholder function here
uint xorshift(uint seed, uint id) {
    uint state = seed + id;
    state ^= state << 16;
    state *= 1000000007; // space out state for consecutive ids
    if (state == 0) { // state must not be 0
        state = 1000000007;
    }
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;

    return state;
}

kernel void rand(
    device float* buf [[buffer(0)]],
    constant uint& seed [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    buf[id] = (float)(xorshift(seed, id)) / (MAXSEEDF + 1); // add 1 to ensure value is not 1.0
}

#define random_int(dtype, name) \
kernel void name( \
    device dtype* buf [[buffer(0)]], \
    constant uint& seed [[buffer(1)]], \
    constant dtype& a [[buffer(2)]], \
    constant dtype& b [[buffer(3)]], \
    uint id [[thread_position_in_grid]] \
) { \
    /* generate random float in range [a, b+1), then round down */ \
    buf[id] = (dtype)((float)(xorshift(seed, id)) / (MAXSEEDF + 1) * (b - a + 1) + a); \
}

#define random_float(dtype, name) \
kernel void name( \
    device dtype* buf [[buffer(0)]], \
    constant uint& seed [[buffer(1)]], \
    constant dtype& a [[buffer(2)]], \
    constant dtype& b [[buffer(3)]], \
    uint id [[thread_position_in_grid]] \
) { \
    buf[id] = (dtype)((float)(xorshift(seed, id)) / MAXSEEDF * (b - a) + a); \
}


// redefine things in the specialize folder here

#define SPECIALIZE_ALL \
TEMPLATE(int32) \
TEMPLATE(float32)

#define SPECIALIZE_NUMERIC \
TEMPLATE(int32) \
TEMPLATE(float32)

#define SPECIALIZE_INT \
TEMPLATE(int32)

#define SPECIALIZE_FLOAT \
TEMPLATE(float32)

// apply functions

#define TEMPLATE(dtype) binary_op(dtype, add_##dtype, +)
SPECIALIZE_ALL
#define TEMPLATE(dtype) binary_op(dtype, sub_##dtype, -)
SPECIALIZE_ALL
#define TEMPLATE(dtype) binary_op(dtype, mul_##dtype, *)
SPECIALIZE_ALL
#define TEMPLATE(dtype) binary_op(dtype, div_##dtype, /)
SPECIALIZE_ALL

#define TEMPLATE(dtype) unary_op(dtype, uadd_##dtype, +=)
SPECIALIZE_ALL
#define TEMPLATE(dtype) unary_op(dtype, usub_##dtype, -=)
SPECIALIZE_ALL
#define TEMPLATE(dtype) unary_op(dtype, umul_##dtype, *=)
SPECIALIZE_ALL
#define TEMPLATE(dtype) unary_op(dtype, udiv_##dtype, /=)
SPECIALIZE_ALL

#define TEMPLATE(dtype) random_int(dtype, random_##dtype)
SPECIALIZE_INT
#define TEMPLATE(dtype) random_float(dtype, random_##dtype)
SPECIALIZE_FLOAT

#undef SPECIALIZE_ALL
#undef SPECIALIZE_NUMERIC
#undef SPECIALIZE_INT
#undef SPECIALIZE_FLOAT
#undef TEMPLATE