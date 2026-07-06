# python things are currently very unsafe
# literally 0 type checking bruh
# TODO: type checking


from . import system, core
from .tensor import *
from .typeutil import *
from typing import Iterable


__all__ = (system, Tensor)


def parse_iterable(depth, first, shape, stack, iterable, result):
    if not isinstance(iterable, Iterable):
        result.append(iterable)
        return shape
    
    for i in iterable:
        stack.append(len(iterable))
        if len(stack) > len(shape):
            if first:
                shape.append(stack[depth])
            else:
                raise RuntimeError("Oxide: depth within iterable does not match")
        elif stack[depth] != shape[depth]:
            raise RuntimeError("Oxide: dimensions do not match")
        parse_iterable(depth + 1, first, shape, stack, i, result)
        stack.pop()
        first = False

    return shape

def tensor(iterable):
    data = []
    shape = parse_iterable(0, True, [], [], iterable, data)
    ctensor = core.make_view(system.backend, shape, data)
    return Tensor(ctensor)

def add(a, b):
    return Tensor(core.binary_add(system.dispatcher, a.ctensor, b.ctensor))

def rand(shape, a = 0, b = 1, d_type = float32):
    if d_type is int32:
        return Tensor(core.rand_int32(system.backend, shape, a, b))
    elif d_type is float32:
        return Tensor(core.rand_float32(system.backend, shape, a, b))
    else:
        raise RuntimeError("Oxide: data type is invalid")