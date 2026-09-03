# Oxide Documentation

Table of contents here

# Setup

### Python

After installing Oxide from [PyPI](https://pypi.org/project/mtl-oxide), import it as shown below:
```py
import oxide
# no further setup needed!
```

[Sample Program](https://github.com/jasononono/oxide/blob/main/docs/sample_py.py) (Python)

### C++

Oxide is a library intended for Python, but the C++ interface is also optionally available. (but more complicated to set up)

To build Oxide for your machine, consider the [CMakeLists.txt](https://github.com/jasononono/oxide/blob/main/CMakeLists.txt) file.

Next, when compiling your program, link the library and its [oxide.hpp](https://github.com/jasononono/oxide/blob/main/src/include/oxide.hpp) header.

In your "main" file:
1. include the [oxide.hpp](https://github.com/jasononono/oxide/blob/main/src/include/oxide.hpp) header
2. create a *run()* function that takes in references for the oxide::Backend& and oxide::Dispatcher& classes.
3. use the 'OXIDE_MAIN' macro after the function to automatically generate the *int main()* entry function.

**Example**
```cpp
#include "oxide.hpp"

void run(oxide::Backend& backend,
         oxide::Dispatcher& dispatcher) {
    // YOUR CODE HERE
}

OXIDE_MAIN
```

[Sample Program](https://github.com/jasononono/oxide/blob/main/docs/sample_cpp.cpp) (C++)

**The documentation onwards will be for the Python interface, but most functions are used similarly in C++.**

# Creating a Tensor

### tensor(iterable)
Create a tensor from a Python iterable.
- **iterable**: the Python iterable (e.g. list, tuple)
```py
my_list = [[1, 3, 5], [2, 4, 6]]
A = oxide.tensor(my_list)
```

### rand(shape)
Generate a tensor filled with random 32-bit floats from the range [0, 1).
- **shape**: shape of the tensor
```py
A = oxide.rand([3, 4])
```

### random(shape, a = 0, b = 1, dtype = float32)
Generate a tensor filled with random numbers from the range [a, b].
- **shape**: shape of the tensor
- **a**: lower bound of the sample range, inclusive
- **b**: upper bound of the sample range, inclusive
- **dtype**: data type of the elements
```py
A = oxide.random([3, 4], 2, 8, oxide.float32)
```

### filled(shape, value, dtype = float32)
Generate a tensor filled with the specified value.
- **shape**: shape of the tensor
- **value**: value to be filled
- **dtype**: data type of the elements
```py
A = oxide.filled([3, 4], 3.14, oxide.float32)
```

### zeros(shape, dtype = float32)
Generate a tensor filled with 0s.
- **shape**: shape of the tensor
- **dtype**: data type of the elements
```py
A = oxide.zeros([3, 4], oxide.float32)
```

### ones(shape, dtype = float32)
Generate a tensor filled with 1s.
- **shape**: shape of the tensor
- **dtype**: data type of the elements
```py
A = oxide.ones([3, 4], oxide.float32)
```

# Tensor Indexing
Elements are retrieved from a tensor with a list of *indices*, where each corresponds to a dimension of the tensor.

Oxide tensors are **zero-indexed**. Negative indices will be treated as if **counting from the right**.

```py
A = oxide.rand([5, 6, 4])

A[3, 2, 3] = 55.12
print(A[3, 2, -1])

print(A[10, -20, 30]) # Error! Index out of range
print(A[3, 2]) # Error! Not enough dimensions
print(A[3][2][3]) # Error! Not formatted correctly
```

# Tensor Class Attributes
The *Tensor* class has many attributes that the user can query.

### Tensor.ndim
Number of dimensions

### Tensor.size
Total number of elements

### Tensor.offset
Offset of view, used to calculate position in the underlying data buffer

### Tensor.shape
Shape of tensor

### Tensor.strides
Strides of view, used to calculate position in the underlying data buffer

### Tensor.dtype
Data type of the tensor's elements

# Tensor Arithmetic

### add(a, b)
Element-wise addition.
```py
out = oxide.add(a, b)
out = a + b
```
Similarly, subtraction (sub), multiplication (mul), and division (div) all work the same.

### uadd(a, b)
Element-wise in-place addition. The result will be stored in the first tensor.
```py
oxide.uadd(a, b)
a += b
```
Similarly, subtraction (usub), multiplication (umul), and division (udiv) all work the same.

# Miscellaneous Functions

### topy(tensor)
Returns the tensor as a python list
- **tensor**: the Oxide tensor
```py
A = oxide.rand([3, 4])
my_list = oxide.topy(A)
print(my_list)
```