#include <metal_stdlib>
using namespace metal;


typedef int32_t int32;
typedef float float32;

#define MAXDIMS 32;


#define binary_op(d_type, name, op) \ // TODO: optimize these for common cases
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


binary_op(int32, add_int32, +)
binary_op(float32, add_float32, +)
binary_op(int32, sub_int32, -)
binary_op(float32, sub_float32, -)
binary_op(int32, mul_int32, *)
binary_op(float32, mul_float32, *)
binary_op(int32, div_int32, /)
binary_op(float32, div_float32, /)

unary_op(int32, uadd_int32, +=)
unary_op(float32, uadd_float32, +=)
unary_op(int32, usub_int32, -=)
unary_op(float32, usub_float32, -=)
unary_op(int32, umul_int32, *=)
unary_op(float32, umul_float32, *=)
unary_op(int32, udiv_int32, /=)
unary_op(float32, udiv_float32, /=)