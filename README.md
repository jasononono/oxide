My attempt at making a GPU-accelerated array computing library for MacOS.

Giving me access to these GPU shenanigans is an equally good idea as installing a GE90 Jet Engine on a scooter.

# Usage

I lowkey linked 2 functions so far, so here is all the functionality

```python
import py_oxide as ox

a = ox.tensor([[1, 2, 3], [4, 5, 6]])
print(a.shape) # [2, 3]
b = ox.tensor([[6, 7, 6], [7, 6, 7]])
print(b[1, 2]) # 7
print(ox.add(a, b)) # [[7, 9, 9], [11, 11, 13]]
c = ox.rand(shape=[6, 7, 8], a=10, b=20, d_type=ox.float32)
```

run this to install:
```
pip install -i https://test.pypi.org/simple/ py-oxide
```

# Features

- auto memory management (alloc and free)
- eager execution
- GPU accelerated "tensors" using MTL::ResourceStorageModeShared
- c++ backend (pain)

# Things

Inspirations: numpy, torch, mlx, and forge (https://github.com/kellen-sun/Forge)

I am unfortunately still trying to comprehend JIT Graph Compilation and like 1000+ other things