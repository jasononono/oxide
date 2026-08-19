# PyOxide

My attempt to build a GPU-accelerated array computing library to learn many things in the process.

Oxide functions similarly to [numpy](https://numpy.org/), built specifically to compute efficiently on the macOS GPU using [Metal](https://developer.apple.com/metal/).

The backend of Oxide is written entirely in C/C++, linked to Python API via [nanobind](https://github.com/wjakob/nanobind/).

# Usage

Oxide currently supports multiple data types, basic arithmetic, and shape manipulation, all optimized with array views and batch processing.

Full documentation coming soon perhaps?

**Sample Program (Python)**

```python
import oxide as ox

a = ox.tensor([[1, 2, 3], [4, 5, 6]])
print(a.shape) # [2, 3]
b = ox.tensor([[6, 7, 6], [7, 6, 7]])
print(b[[1, 2]]) # 7    (use lists to index tensors)
print(ox.add(a, b)) # [[7, 9, 9], [11, 11, 13]]
c = a # efficiently copies view 'a' to 'c'
d = ox.random(shape=[6, 7, 8], a=10, b=20, dtype=ox.float32)
```

# Platform & Installation

Oxide is a macOS-exclusive library.

Run this to install the [TestPyPI](https://test.pypi.org/simple/) distribution:

```
pip install -i https://test.pypi.org/simple/ py-oxide
```

# Features

- GPU accelerated "tensors" using MTL::ResourceStorageModeShared
- automatic Tensor memory management (alloc and free)
- c++ backend (pain)
- GPU pseudo-random generator (xorshift currently a little scuffed)

# Inspirations
Parsing through these awesome libraries helped me setup Metal, design the file system, and understand the complicated shader syntax!

- [numpy](https://numpy.org/): most oxide function names corresponds to numpy
- [torch](https://pytorch.org/): benchmarking alongside numpy
- [forge](https://github.com/kellen-sun/Forge/): Metal library setup reference