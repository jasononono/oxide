from .common import *
from . import system
from . import util
from .tensorclass import *


def rand(shape):
    shape = util.check_shape(shape)
    return Tensor(system.run("rand", system.dispatcher, shape))

def random(shape, a = 0, b = 1, dtype = float32):
    shape = util.check_shape(shape)
    util.check_dtype(dtype, numeric_t)
    util.check_args(dtype, a, b)

    return Tensor(system.run(with_type(dtype, "random"), system.dispatcher, shape, a, b))

def filled(shape, value, dtype = float32):
    shape = util.check_shape(shape)
    util.check_dtype(dtype, oxide_t)
    util.check_args(dtype, value)

    return Tensor(system.run(with_type(dtype, "filled"), system.backend, shape, value))

def zeros(shape, dtype = float32):
    shape = util.check_shape(shape)
    util.check_dtype(dtype, numeric_t)

    return Tensor(system.run(with_type(dtype, "zeros"), system.backend, shape))

def ones(shape, dtype = float32):
    shape = util.check_shape(shape)
    util.check_dtype(dtype, numeric_t)

    return Tensor(system.run(with_type(dtype, "ones"), system.backend, shape))


# def transpose(view, order):
#     return Tensor(system.run(core.transpose, view.ctensor, order))