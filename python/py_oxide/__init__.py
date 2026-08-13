from .common import *
from . import system
from . import core
from .tensor import *
from typing import Iterable


__all__ = (system, Tensor)


def parse_iterable(depth, first, shape, stack, iterable, result):
    if not isinstance(iterable, Iterable):
        for t in pyconv_t.keys():
            if isinstance(iterable, t):
                result.append(iterable)
                return pyconv_t[t]
        system.throw(f"item of type '{type(iterable).__name__}' cannot be stored in a tensor")

    t = None
    if len(iterable) == 0:
        system.throw("empty iterable cannot be a part of tensor")

    for i in iterable:
        stack.append(len(iterable))
        if len(stack) > len(shape):
            if first:
                shape.append(stack[depth])
            else:
                system.throw("depth within iterable does not match")
        elif stack[depth] != shape[depth]:
            system.throw("dimensions do not match")
            
        t_new = parse_iterable(depth + 1, first, shape, stack, i, result)
        if t and t_new is not t:
            system.throw("tensor can only store items of similar types")
        t = t_new

        stack.pop()
        first = False

    return t

def tensor(iterable):
    data = []
    shape = []
    parse_iterable(0, True, shape, [], iterable, data)
    ctensor = system.run(core.make_view, system.backend, shape, data)
    return Tensor(ctensor)


def add(a, b):
    if not (isinstance(a, Tensor) and isinstance(b, Tensor)):
        system.throw("tensor addition expected tensor arguments")
    if a.dtype != b.dtype:
        system.throw("tensor addition expected tensors of the same type")
    return Tensor(system.run(core.binary_add, system.dispatcher, a.ctensor, b.ctensor))


# def rand(shape):
#     return Tensor(core.rand(system.dispatcher, shape))

# def random(shape, a = 0, b = 1, d_type = float32):
#     if d_type is int32:
#         return Tensor(core.random_int32(system.dispatcher, shape, a, b))
#     elif d_type is float32:
#         return Tensor(core.random_float32(system.dispatcher, shape, a, b))
#     else:
#         raise RuntimeError("Oxide: data type is invalid")