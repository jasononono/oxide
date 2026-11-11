"""
VIEWMANIP.PY

tensor shape manipulation functions
"""


from .common import *
from . import system
from . import util
from .tensorclass import *


# returns a reshaped view without modifying the data (whenever possible). AVOID and FORCED flags are used as copy options
def reshape(view, shape, copy = 0):
    util.check_tensors(view)
    shape = util.check_shape(shape)
    return Tensor(system.run("reshape", system.dispatcher, view.ctensor, shape, copy))

# returns a flattened 1d view. AVOID and FORCED flags are used as copy options
def ravel(view, copy = 0):
    util.check_tensors(view)
    return Tensor(system.run("ravel", system.dispatcher, view.ctensor, copy))

# what ravel() does, but modifies the view in-place (if possible)
def flatten(view):
    util.check_tensors(view)
    return Tensor(system.run("flatten", view.ctensor))

# reorder the axes
def transpose(view, order):
    util.check_tensors(view)
    order = util.check_iterable(int, order, True)
    return Tensor(system.run("transpose", view.ctensor, order))