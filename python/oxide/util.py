"""
UTIL.PY

utility functions (type & input checkers)
"""


from .common import *
from . import system
from .tensorclass import *
import collections


# ensure tensor types are univariant
# set cast_consts to True to convert all constants into 0d tensors
def check_tensors(*tensors, cast_consts = False):
    dtypes = set()
    for t in tensors:
        if isinstance(t, Tensor):
            dtypes.add(t.dtype)
        elif not cast_consts:
            system.throw(f"argument should be tensor, not '{type(t).__name__}'")
        
    if len(dtypes) > 1:
        system.throw("tensors should be of the same type")
    dtype = list(dtypes)[0]

    if cast_consts:
        new_tensors = []
        for t in tensors:
            if isinstance(t, Tensor):
                new_tensors.append(t)
            else:
                nt = system.viewdata_module.constview(t, dtype)
                if nt.dtype is not dtype:
                    system.throw("constant dtype does match tensor dtypes")
                new_tensors.append(nt)
        return dtype, new_tensors
    
    return dtype, tensors

# ensure that tensor shape is valid
def check_shape(shape):
    if isinstance(shape, int):
        shape = [shape]
    if len(shape) > MAXDIMS:
        system.throw("shape exceeds maximum dimensions")
    for i in shape:
        if not (isinstance(i, int) and i > 0):
            system.throw("shape is invalid")
    return shape

# ensure that dtype is an oxide dtype
# optionally stricter restrictions
def check_dtype(dtype, tclass = None):
    if tclass is None:
        if not issubclass(dtype, oxide_t):
            system.throw("dtype is invalid")
    else:
        if not issubclass(dtype, tclass):
            system.throw("dtype is invalid")

def check_args(dtype, *args):
    for a in args:
        if not isinstance(a, OX_T[dtype]):
            system.throw("argument type is invalid")

# check an iterable for elements of dtype, optionally enforced unsignedness
def check_iterable(dtype, it, unsigned = False):
    if not isinstance(it, collections.abc.Sequence):
        system.throw("argument must be an iterable")
    it = list(it)
    for i in it:
        if not isinstance(i, dtype):
            system.throw("iterable elements type invalid")
        if unsigned and i < 0:
            system.throw("iterable elements must be unsigned")
    return it