from .common import *
from . import system
from .tensor import Tensor, tensor
from .operands import *
from . import core


def check_shape(shape):
    if isinstance(shape, int):
        shape = [shape]
    if len(shape) > MAXDIMS:
        system.throw("shape exceeds maximum dimensions")
    for i in shape:
        if not (isinstance(i, int) and i > 0):
            system.throw("shape is invalid")
    return shape


def rand(shape):
    shape = check_shape(shape)
    return Tensor(system.run(core.rand, system.dispatcher, shape))

def random(shape, a = 0, b = 1, dtype = float32):
    shape = check_shape(shape)

    if not dtype in OX_T:
        system.throw("dtype is invalid")
    if not (isinstance(a, OX_T[dtype]) and isinstance(b, OX_T[dtype])):
        system.throw("random range argument types are invalid")
    return Tensor(system.run(core.random, system.dispatcher, shape, a, b))


def filled(shape, value, dtype = float32):
    shape = check_shape(shape)

    if not dtype in OX_T:
        system.throw("dtype is invalid")
    if not isinstance(value, OX_T[dtype]):
        system.throw("value type is invalid")
    return Tensor(system.run(core.filled, system.backend, shape, value))

def zeros(shape, dtype = float32):
    shape = check_shape(shape)

    if not dtype in OX_T:
        system.throw("dtype is invalid")
    return Tensor(system.run(core.zeros, system.backend, shape))

def ones(shape, dtype = float32):
    shape = check_shape(shape)

    if not dtype in OX_T:
        system.throw("dtype is invalid")
    return Tensor(system.run(core.ones, system.backend, shape))


def transpose(view, order):
    return Tensor(system.run(core.transpose, view.ctensor, order))