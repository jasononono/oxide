"""
__INIT__.PY

module init
"""


from .common import *
from . import system
from . import util
from .tensorclass import *
from .arithmetic import *
from .viewdata import *
from .viewmanip import *
from . import core


# these module references are stored at runtime to avoid circular import
system.arithmetic_module = arithmetic
system.viewdata_module = viewdata