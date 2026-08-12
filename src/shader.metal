#include <metal_stdlib>
using namespace metal;


// common.hpp
using uint = uint32_t;
using iint = int32_t;

typedef int32_t int32;
typedef float float32;

#define MAXDIMS 32
#define MAXSEEDF 4294967295.0f


// functions
// TODO: optimize these for common cases
#define binary_op(d_type, name, op) \
kernel void name( \
    const device d_type* a [[buffer(0)]], \
    const device d_type* b [[buffer(1)]], \
    device d_type* out [[buffer(2)]], \
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
    for (uint i = 0; i < ndim; i++) { \
        coord = out_idx / out_strides[i]; \
        out_idx %= out_strides[i]; \
        a_idx += coord * a_strides[i]; \
        b_idx += coord * b_strides[i]; \
    } \
\
    out[id] = a[a_idx] op b[b_idx]; \
}

#define unary_op(d_type, name, op) \
kernel void name( \
    device d_type* a [[buffer(0)]], \
    const device d_type* b [[buffer(1)]], \
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
    for (uint i = 0; i < ndim; i++) { \
        coord = a_idx / a_strides[i]; \
        a_idx %= a_strides[i]; \
        b_idx += coord * b_strides[i]; \
    } \
\
    a_idx += id + a_offset; \
    a[a_idx] op b[b_idx]; \
}


uint xorshift(uint seed, uint id) {
    uint state = seed + id;
    state ^= state << 16;
    state *= 2994277826;
    if (state == 0) {
        state = 2994277826;
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
    buf[id] = (float)(xorshift(seed, id)) / (MAXSEEDF + 1);
}

#define random_int(d_type, name) \
kernel void name( \
    device d_type* buf [[buffer(0)]], \
    constant uint& seed [[buffer(1)]], \
    constant d_type& a [[buffer(2)]], \
    constant d_type& b [[buffer(3)]], \
    uint id [[thread_position_in_grid]] \
) { \
    buf[id] = (d_type)((float)(xorshift(seed, id)) / (MAXSEEDF + 1) * (b - a + 1) + a); \
}

#define random_float(d_type, name) \
kernel void name( \
    device d_type* buf [[buffer(0)]], \
    constant uint& seed [[buffer(1)]], \
    constant d_type& a [[buffer(2)]], \
    constant d_type& b [[buffer(3)]], \
    uint id [[thread_position_in_grid]] \
) { \
    buf[id] = (d_type)((float)(xorshift(seed, id)) / MAXSEEDF * (b - a) + a); \
}


// apply functions
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

#define TEMPLATE(d_type) binary_op(d_type, add_##d_type, +)
SPECIALIZE_ALL
#define TEMPLATE(d_type) binary_op(d_type, sub_##d_type, -)
SPECIALIZE_ALL
#define TEMPLATE(d_type) binary_op(d_type, mul_##d_type, *)
SPECIALIZE_ALL
#define TEMPLATE(d_type) binary_op(d_type, div_##d_type, /)
SPECIALIZE_ALL

#define TEMPLATE(d_type) unary_op(d_type, uadd_##d_type, +=)
SPECIALIZE_ALL
#define TEMPLATE(d_type) unary_op(d_type, usub_##d_type, -=)
SPECIALIZE_ALL
#define TEMPLATE(d_type) unary_op(d_type, umul_##d_type, *=)
SPECIALIZE_ALL
#define TEMPLATE(d_type) unary_op(d_type, udiv_##d_type, /=)
SPECIALIZE_ALL

#define TEMPLATE(d_type) random_int(d_type, random_##d_type)
SPECIALIZE_INT
#define TEMPLATE(d_type) random_float(d_type, random_##d_type)
SPECIALIZE_FLOAT

#undef SPECIALIZE_ALL
#undef SPECIALIZE_NUMERIC
#undef SPECIALIZE_INT
#undef SPECIALIZE_FLOAT
#undef TEMPLATE