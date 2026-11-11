"""
VIEWDATA.PY

tensor initialization, copying, and randomize functions
"""


from .common import *
from . import system
from . import util
from .tensorclass import *


# copy the contents of a view to another using the gpu
def set_view(view, value):
    dtype, (view, value) = util.check_tensors(view, value, cast_consts = True)
    system.run(with_type(dtype, "set_view"), system.dispatcher, view.ctensor, value.ctensor)    

# generate float32 tensor randomly filled with elements from range [0, 1)
def rand(shape):
    shape = util.check_shape(shape)
    return Tensor(system.run("rand", system.dispatcher, shape))

# generate tensor randomly filled with elements from range [a, b]
def random(shape, a = 0, b = 1, dtype = float32):
    shape = util.check_shape(shape)
    util.check_dtype(dtype, numeric_t)
    util.check_args(dtype, a, b)

    return Tensor(system.run(with_type(dtype, "random"), system.dispatcher, shape, a, b))

# generate tensor filled with value
def filled(shape, value, dtype = float32):
    shape = util.check_shape(shape)
    util.check_dtype(dtype)
    util.check_args(dtype, value)

    return Tensor(system.run(with_type(dtype, "filled"), system.backend, shape, value))

# generate tensor filled with 0s
def zeros(shape, dtype = float32):
    shape = util.check_shape(shape)
    util.check_dtype(dtype, numeric_t)

    return Tensor(system.run(with_type(dtype, "zeros"), system.backend, shape))

# generate tensor filled with 1s
def ones(shape, dtype = float32):
    shape = util.check_shape(shape)
    util.check_dtype(dtype, numeric_t)

    return Tensor(system.run(with_type(dtype, "ones"), system.backend, shape))

# convert value into 0d tensor
def constview(value, dtype = float32):
    util.check_dtype(dtype)
    util.check_args(dtype, value)
    return Tensor(system.run(with_type(dtype, "constview"), system.backend, value))

