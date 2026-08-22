from .common import *
from . import system
from .tensor import *


def check_tensors(*tensors):
    dtypes = set()
    for t in tensors:
        if not isinstance(t, Tensor):
            system.throw(f"argument should be tensor, not '{type(t).__name__}'")
        dtypes.add(t.dtype)

    if len(dtypes) > 1:
        system.throw("tensors should be of the same type")
    return list(dtypes)[0]

def check_shape(shape):
    if isinstance(shape, int):
        shape = [shape]
    if len(shape) > MAXDIMS:
        system.throw("shape exceeds maximum dimensions")
    for i in shape:
        if not (isinstance(i, int) and i > 0):
            system.throw("shape is invalid")
    return shape

def check_dtype(dtype, tclass = None):
    if tclass is None:
        if not dtype in OX_T:
            system.throw("dtype is invalid")
    else:
        if not issubclass(dtype, tclass):
            system.throw("dtype is invalid")

def check_args(dtype, *args):
    for a in args:
        if not isinstance(a, OX_T[dtype]):
            system.throw("argument type is invalid")