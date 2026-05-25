#include <metal_stdlib>
using namespace metal;


#define binary_op(d_type, name, op) \
kernel void name( \
    const device d_type* a [[buffer(0)]], \
    const device d_type* b [[buffer(1)]], \
    device d_type* out [[buffer(2)]], \
    uint id [[thread_position_in_grid]] \
) { \
    out[id] = a[id] op b[id]; \
}

#define unary_op(d_type, name, op) \
kernel void name( \
    device d_type* a [[buffer(0)]], \
    const device d_type* b [[buffer(1)]], \
    uint id [[thread_position_in_grid]] \
) { \
    a[id] op b[id]; \
}

typedef int32_t int32;
typedef float float32;


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