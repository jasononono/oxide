from .common import *
from . import system
from . import core
import typing


class Tensor:
    def __init__(self, ctensor):
        super().__setattr__("ctensor", ctensor)
        for t in tensorview_t.keys():
            if isinstance(ctensor, t):
                super().__setattr__("dtype", tensorview_t[t])
                break
        else:
            system.throw("ctensor argument type invalid")

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.ctensor.get_element([key])
        elif isinstance(key, typing.Iterable):
            return self.ctensor.get_element(list(key))
        else:
            system.throw(f"tensor cannot be indexed with '{type(key).__name__}'")

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.ctensor.set_element([key], value)
        elif isinstance(key, typing.Iterable):
            self.ctensor.set_element(list(key), value)
        else:
            system.throw(f"tensor cannot be indexed with '{type(key).__name__}'")

    def __getattr__(self, key):
        if key == "ndim":
            return self.ctensor.get_ndim()
        if key == "size":
            return self.ctensor.get_size()
        if key == "offset":
            return self.ctensor.get_offset()
        if key == "shape":
            return self.ctensor.get_shape().copy()
        if key == "strides":
            return self.ctensor.get_strides().copy()
        if key == "dtype":
            return self.dtype
        
        system.throw(f"tensor attribute '{key}' does not exist")
        
    def __setattr__(self, key, value):
        system.throw("tensor attributes are immutable")
    
    def __str__(self):
        return self.ctensor.get_string()