from .common import *
from . import system
from . import util
from .tensorclass import *


def add(a, b):
    dtype = util.check_tensors(a, b)
    return Tensor(system.run(with_type(dtype, "add"), system.dispatcher, a.ctensor, b.ctensor))

def sub(a, b):
    dtype = util.check_tensors(a, b)
    return Tensor(system.run(with_type(dtype, "sub"), system.dispatcher, a.ctensor, b.ctensor))

def mul(a, b):
    dtype = util.check_tensors(a, b)
    return Tensor(system.run(with_type(dtype, "mul"), system.dispatcher, a.ctensor, b.ctensor))

def div(a, b):
    dtype = util.check_tensors(a, b)
    return Tensor(system.run(with_type(dtype, "div"), system.dispatcher, a.ctensor, b.ctensor))


def uadd(a, b):
    dtype = util.check_tensors(a, b)
    return Tensor(system.run(with_type(dtype, "uadd"), system.dispatcher, a.ctensor, b.ctensor))

def usub(a, b):
    dtype = util.check_tensors(a, b)
    return Tensor(system.run(with_type(dtype, "usub"), system.dispatcher, a.ctensor, b.ctensor))

def umul(a, b):
    dtype = util.check_tensors(a, b)
    return Tensor(system.run(with_type(dtype, "umul"), system.dispatcher, a.ctensor, b.ctensor))

def udiv(a, b):
    dtype = util.check_tensors(a, b)
    return Tensor(system.run(with_type(dtype, "udiv"), system.dispatcher, a.ctensor, b.ctensor))