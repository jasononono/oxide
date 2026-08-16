from .common import *
from . import system
from .tensor import Tensor, tensor
from . import core


def add(a, b):
    if not (isinstance(a, Tensor) and isinstance(b, Tensor)):
        system.throw("tensor addition expected tensor arguments")
    if a.dtype != b.dtype:
        system.throw("tensor addition expected tensors of the same type")
    return Tensor(system.run(core.binary_add, system.dispatcher, a.ctensor, b.ctensor))

def check_shape(shape):
    if len(shape) > MAXDIMS:
        system.throw("shape exceeds maximum dimensions")
    for i in shape:
        if not (isinstance(i, int) and i > 0):
            system.throw("shape is invalid")

def rand(shape):
    if isinstance(shape, int):
        shape = [shape]
    check_shape(shape)

    return Tensor(system.run(core.rand, system.dispatcher, shape))

def random(shape, a = 0, b = 1, dtype = float32):
    if isinstance(shape, int):
        shape = [shape]
    check_shape(shape)

    if not dtype in OX_T:
        system.throw("dtype is invalid")
    if not (isinstance(a, OX_T[dtype]) and isinstance(b, OX_T[dtype])):
        system.throw("random range argument types are invalid")
    return Tensor(system.run(core.random, system.dispatcher, shape, a, b))

def filled(shape, value, dtype = float32):
    if isinstance(shape, int):
        shape = [shape]
    check_shape(shape)

    if not dtype in OX_T:
        system.throw("dtype is invalid")
    if not isinstance(value, OX_T[dtype]):
        system.throw("value type is invalid")
    return Tensor(system.run(core.filled, system.backend, shape, value))

def zeros(shape, dtype = float32):
    if isinstance(shape, int):
        shape = [shape]
    check_shape(shape)

    if not dtype in OX_T:
        system.throw("dtype is invalid")
    return Tensor(system.run(core.zeros, system.backend, shape))