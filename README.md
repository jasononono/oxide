# About Oxide

My attempt at building a **GPU-accelerated array computing library** to learn many things in the process.

**Oxide** functions similarly to [Numpy](https://numpy.org/), built exclusively to compute efficiently on the **macOS GPU** using [Metal](https://developer.apple.com/metal/).

The backend of Oxide is written entirely in **C/C++**, linked to Python interface via [nanobind](https://github.com/wjakob/nanobind/).

# Usage

Oxide currently supports multiple data types, basic arithmetic, and shape manipulation, all optimized with array views and batch processing.

[Documentation](https://github.com/jasononono/oxide/blob/main/docs/DOC.md) \
[Code Explanation & Notes](https://github.com/jasononono/oxide/blob/main/docs/DEV.md) \
[Benchmarks](https://github.com/jasononono/oxide/blob/main/docs/BENCHMARK.md)

# Platform & Installation

Oxide is a **macOS-exclusive library** (your device must also have a gpu because I haven't implement a cpu fallback yet)

Run this command to install the latest [PyPI](https://pypi.org/project/mtl-oxide) distribution:

```
pip install mtl-oxide
```

# Features

- GPU accelerated *Tensor* class using MTL::ResourceStorageModeShared
- automatic Tensor memory management (alloc, free, and optimization)
- memory safe c++ backend (no leaks detected)
- GPU pseudo-random generator (xorshift currently a little scuffed, ill replace it soon)

# Inspirations
Parsing through these awesome libraries helped me setup Metal, design the file system, and understand the complicated shader syntax!

- [numpy](https://numpy.org/): most oxide function names corresponds to numpy
- [torch](https://pytorch.org/): benchmarking alongside numpy
- [forge](https://github.com/kellen-sun/Forge/): Metal library setup reference

# Contributing
Oxide is my personal learning project, but feel free to open an issue on [Github](https://github.com/jasononono/oxide) if you have any suggestions!