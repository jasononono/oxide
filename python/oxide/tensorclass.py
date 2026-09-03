"""
TENSORCLASS.PY

tensor class definition and conversion functions
"""


from .common import *
from . import system
from . import core
import typing


class Tensor:

    def __init__(self, ctensor):
        super().__setattr__("ctensor", ctensor)
        # match class name with dtype
        for t in OX_T:
            ct =  with_type(t, "TensorView")
            if ct == type(ctensor).__name__:
                super().__setattr__("dtype", t)
                break
        else:
            system.throw("ctensor type invalid")

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

    def __add__(self, other):
        return system.operand_module.add(self, other)

    def __sub__(self, other):
        return system.operand_module.sub(self, other)

    def __mul__(self, other):
        return system.operand_module.mul(self, other)

    def __truediv__(self, other):
        return system.operand_module.div(self, other)


# convert oxide tensor into python list
def topy(tensor):

    def generate(depth = 0): # generate an empty list with the tensor's shape
        if depth == tensor.ndim:
            return 0
        arr = [generate(depth + 1) for _ in range(tensor.shape[depth])]
        return arr
    
    arr = generate()

    arr_tree = [] # current dfs stack of list references
    ref = arr
    while ref != 0:
        arr_tree.append(ref)
        ref = ref[0]

    indices = [0] * tensor.ndim

    # dfs traversal, similar implementation to get_string() function in tensor.cpp
    while len(indices) > 0:
        arr_tree[-1][indices[-1]] = pycast(tensor.dtype, tensor[indices])

        while len(indices) > 0 and indices[-1] == tensor.shape[len(indices) - 1] - 1:
            indices.pop(-1)
            arr_tree.pop(-1)

        if len(indices) == 0:
            break

        indices[-1] += 1
        if len(indices) != tensor.ndim:
            # append new ref and recursively append its children
            arr_tree.append(arr_tree[-1][indices[-1]])
            indices.append(0)
            while len(indices) < tensor.ndim:
                arr_tree.append(arr_tree[-1][0])
                indices.append(0)

    return arr
    

############# OUTDATED FUNCTION(s)

# dfs helper function
def parse_iterable(depth, first, shape, stack, iterable, result):
    if not isinstance(iterable, typing.Iterable):
        # constant must be able to be convert into a valid oxide dtype
        for t in PY_T.keys():
            if isinstance(iterable, t):
                result.append(iterable)
                return PY_T[t]
        system.throw(f"item of type '{type(iterable).__name__}' cannot be stored in a tensor")

    t = None
    if len(iterable) == 0:
        system.throw("empty iterable cannot be a part of tensor")

    for i in iterable:
        stack.append(len(iterable))
        if len(stack) > len(shape):
            if first: # shape is invalid if the maximum encountered depth increases at a position other than the first
                shape.append(stack[depth])
            else:
                system.throw("depth within iterable does not match")
        elif stack[depth] != shape[depth]:
            system.throw("dimensions do not match")

        # types must be univariant
        t_new = parse_iterable(depth + 1, first, shape, stack, i, result)
        if t and t_new is not t:
            system.throw("tensor can only store items of similar types")
        t = t_new

        stack.pop()
        first = False

    return t

# create an oxide tensor from python iterable
def tensor(iterable):
    data = [] # accumulate linear buffer from iterable
    shape = []
    t = parse_iterable(0, True, shape, [], iterable, data)
    ctensor = system.run(with_type(t, "make_view"), system.backend, shape, data)
    return Tensor(ctensor)

#############