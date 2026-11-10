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