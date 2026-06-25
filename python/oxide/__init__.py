from . import system
from .tensor import *
from typing import Iterable


__all__ = (system, Tensor)


def parse_iterable(array):
    stack = []
    current = array
    while isinstance(current[0], Iterable):
        

    while len(stack) > 0:
        


def tensor(iterable):
    
    # iterate through iterable here blah blah blah
    ctensor = None
    return Tensor(ctensor)