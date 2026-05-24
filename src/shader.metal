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
} \


binary_op(float, add_int32, +)
binary_op(float, add_float32, +)