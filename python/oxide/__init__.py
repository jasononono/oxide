from . import system, core
from .tensor import *
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



# python things are currently very unsafe
# literally 0 type checking bruh

a = tensor([1, 2, 3])
b = tensor([2, 3, 4])
print(a, b)
print(add(a, b))