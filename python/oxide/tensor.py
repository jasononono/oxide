from .common import *
from . import system
from . import core
import typing


class Tensor:
    def __init__(self, ctensor):
        super().__setattr__("ctensor", ctensor)
        for t in TENSOR_T.keys():
            if isinstance(ctensor, t):
                super().__setattr__("dtype", TENSOR_T[t])
                break
        else:
            system.throw("ctensor argument type invalid")

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.ctensor.get_element([key])
        elif isinstance(key, typing.Iterable):
            return self.ctensor.get_element(list(key))
        else:
            system.throw(f"tensor cannot be indexed with '{type(key).__name__}'")

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.ctensor.set_element([key], value)
        elif isinstance(key, typing.Iterable):
            self.ctensor.set_element(list(key), value)
        else:
            system.throw(f"tensor cannot be indexed with '{type(key).__name__}'")

    def __getattr__(self, key):
        if key == "ndim":
            return self.ctensor.get_ndim()
        if key == "size":
            return self.ctensor.get_size()
        if key == "offset":
            return self.ctensor.get_offset()
        if key == "shape":
            return self.ctensor.get_shape().copy()
        if key == "strides":
            return self.ctensor.get_strides().copy()
        if key == "dtype":
            return self.dtype
        
        system.throw(f"tensor attribute '{key}' does not exist")
        
    def __setattr__(self, key, value):
        system.throw("tensor attributes are immutable")
    
    def __str__(self):
        return self.ctensor.get_string()


def parse_iterable(depth, first, shape, stack, iterable, result):
    if not isinstance(iterable, typing.Iterable):
        for t in PY_T.keys():
            if isinstance(iterable, t):
                result.append(iterable)
                return PY_T[t]
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