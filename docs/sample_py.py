import sys, os
sys.path.append(os.getcwd())
from python import oxide as ox


# initialize tensor A as a 3x4 matrix of random 32 bit floats in the range [2, 8]
A = ox.random([3, 4], a = 2, b = 8, dtype = ox.float32)
# initialize tensor B as a 3x4 matrix filled with the value 3.14 (as 32 bit floats)
B = ox.filled([3, 4], value = 3.14, dtype = ox.float32)
# initialize tensor C from a python iterable
C = ox.tensor([[1, 3, 5], [2, 4, 6]])

# display tensor A formatted nicely
print(A)
# retrieve an element of tensor A using a list/tuple of indices
print(A[0, 2])
# you can also access more information about the view, a few examples are given below:
print(A.ndim, A.size) # # of dimensions, total # of elements
print(A.shape) # shape of tensor A
print()

OUT = A + B # perform element-wise addition on A and B (out = a + b)
# the above line is equivalent to OUT = ox.add(A, B)
print(OUT)

A *= B # multiply element-wise in place
# the above line is equivalent to ox.umul(A, B)
print(A)

t = ox.topy(A) # convert tensor A into a python list