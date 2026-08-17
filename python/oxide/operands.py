from .common import *
from . import system
from .tensor import Tensor
from . import core


def add(a, b):
    if not (isinstance(a, Tensor) and isinstance(b, Tensor)):
        system.throw("tensor addition expected tensor arguments")
    if a.dtype != b.dtype:
        system.throw("tensor addition expected tensors of the same type")
    return Tensor(system.run(core.binary_add, system.dispatcher, a.ctensor, b.ctensor))

def sub(a, b):
    if not (isinstance(a, Tensor) and isinstance(b, Tensor)):
        system.throw("tensor subtraction expected tensor arguments")
    if a.dtype != b.dtype:
        system.throw("tensor subtraction expected tensors of the same type")
    return Tensor(system.run(core.binary_sub, system.dispatcher, a.ctensor, b.ctensor))

def mul(a, b):
    if not (isinstance(a, Tensor) and isinstance(b, Tensor)):
        system.throw("tensor multiplication expected tensor arguments")
    if a.dtype != b.dtype:
        system.throw("tensor multiplication expected tensors of the same type")
    return Tensor(system.run(core.binary_mul, system.dispatcher, a.ctensor, b.ctensor))

def div(a, b):
    if not (isinstance(a, Tensor) and isinstance(b, Tensor)):
        system.throw("tensor division expected tensor arguments")
    if a.dtype != b.dtype:
        system.throw("tensor division expected tensors of the same type")
    return Tensor(system.run(core.binary_div, system.dispatcher, a.ctensor, b.ctensor))


def uadd(a, b):
    if not (isinstance(a, Tensor) and isinstance(b, Tensor)):
        system.throw("tensor addition expected tensor arguments")
    if a.dtype != b.dtype:
        system.throw("tensor addition expected tensors of the same type")
    return Tensor(system.run(core.unary_add, system.dispatcher, a.ctensor, b.ctensor))

def usub(a, b):
    if not (isinstance(a, Tensor) and isinstance(b, Tensor)):
        system.throw("tensor subtraction expected tensor arguments")
    if a.dtype != b.dtype:
        system.throw("tensor subtraction expected tensors of the same type")
    return Tensor(system.run(core.unary_sub, system.dispatcher, a.ctensor, b.ctensor))

def umul(a, b):
    if not (isinstance(a, Tensor) and isinstance(b, Tensor)):
        system.throw("tensor multiplication expected tensor arguments")
    if a.dtype != b.dtype:
        system.throw("tensor multiplication expected tensors of the same type")
    return Tensor(system.run(core.unary_mul, system.dispatcher, a.ctensor, b.ctensor))

def udiv(a, b):
    if not (isinstance(a, Tensor) and isinstance(b, Tensor)):
        system.throw("tensor division expected tensor arguments")
    if a.dtype != b.dtype:
        system.throw("tensor division expected tensors of the same type")
    return Tensor(system.run(core.unary_div, system.dispatcher, a.ctensor, b.ctensor))