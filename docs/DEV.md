# Oxide Developer Documentation

This is a documentation of Oxide's internal implementation and design pattern.

(mostly an unsorted collection of the library's unique features)

# Oxide's layers of abstraction

I tried to group all functions created and used in Oxide into distinct **layers of abstraction**:

    - L0: shader program (direct GPU communication)
    - L1: backend: handles, random generators, memory management, etc.
    - L2: dispatcher: object that uses backend to configure the GPU and perform arbitrary computations
    - L3: functions & tensors: provides a usable c++ interface and perform basic checks (e.g. no negative dimensions)
    - L4: py-oxide: python interface, performs more checks (types, python-specific implementation)

Each layer acts as units that function independently, so changing the implementation of *Backend* won't affect how the *Dispatcher* uses the GPU pipeline. This design pattern also helps limiting the scope of functions you would need to consider during debugging.

# Data type specialization

Functions that accept tensors of multiple data types always use C++'s template specialization pattern. The template definition is placed in the .hpp file, implemented in the .cpp file, and finally specialized for each Oxide dtype right after the implementation.

Here is what such a function might look like (.cpp file):
```cpp
template <typename dtype>
void some_func(TensorView<dtype>& a, TensorView<dtype>& b) {
    // do some stuff
}
// the function must be defined for each applicable dtype at compile time
// define a macro to simplify each specialization
#define TEMPLATE(dtype) template void some_func(TensorView<dtype>& a, TensorView<dtype>& b);
// automatically use the macro for each oxide dtype
#include "specialize/all.h"
```

The #include statement at the end looks something like ...
```cpp
TEMPLATE(int32)
TEMPLATE(float32)
// do this for more types, and then undef TEMPLATE
```
and is pasted directly into the .cpp program before compilation. Then, the *TEMPLATE* macro is expanded.

The "specialization headers" can also optionally call the macro for every pairing of two types of attributes (e.g. arithmetic operator & dtype), which looks something like this:
```cpp
// turn on "2-dimensional" specialization
#define SPEC2D
// outer most macro, called by the spec header for every dtype
#define TEMPLATE2D(dtype) \
TEMPLATE(add, dtype) \
TEMPLATE(sub, dtype) \
TEMPLATE(mul, dtype) \
TEMPLATE(div, dtype)

// repeat a single operator's implementation
#define TEMPLATE(op, null) template <typename dtype> \
void op() { \
    /* perform arithmetic things */ \
}
TEMPLATE2D(null)

#undef TEMPLATE
#define TEMPLATE(op, dtype) template void op();
// paste TEMPLATE2D for each dtype
#include "specialize/numeric.h"

#undef SPEC2D
```
```cpp
// specialize/numeric.h

#ifdef SPEC2D
    #define MACRO TEMPLATE2D
#else
    #define MACRO TEMPLATE
#endif

MACRO(int32)
MACRO(float32)
// more types, and then undef
```
Note: I still haven't figured out how to include files from shader.metal, so a *SPECIALIZE* macro is used instead

# Data type conventions
Functions that require specialization outside of C++ use the *{name}_{dtype}* naming convention.

Examples include:
- *Tensorview_int32* bound to Python
- *add_float32* bound to Python
- *umul_float32* in shader.metal

When adding a new Oxide dtype, the following places must be modified:

1. all "common variables" locations (see below)
2. specialize headers

# Common variables
When defining a global constant, dtype, or other "common" instances, the following places must all receive that change:

1. common.hpp (C++)
2. common.py (Python)
3. shader.metal (Metal, C++17 syntax)

# Errors and Warnings
When Oxide encounters a fault, it will either throw an error (immediate termination) or log a warning to stdout.

Most errors and warnings are cached in the *error_log* and *warning_log* attribute of *Backend*, only to be displayed in batches at specific times. However, some can be forwarded and outputted directly depending on the interface.

|  | Python | C++ | Description
| --- | --- | --- | --- |
| log warnings | logw() | logw() | displays warning. If C++ functions are run from Python, warnings are appended to *warning_log* and are displayed after the *run()* call.
| flush warnings | flush() | flush() | display and clear all warnings in *warning_log* |
| log errors | - | log(), log_metal() | append errors to *error_log*
| throw errors | throw() | abort() | throw errors. *error_log* is used to route error messages to Python

# GPU randomization
Xorshift acts as a placeholder for a future PRNG algorithm.

# Design references

[Numpy broadcasting](https://numpy.org/devdocs/user/basics.broadcasting.html) \
[Numpy views](https://numpy.org/devdocs/user/basics.copies.html) \
[Metal syntax](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu) \
[Xorshift](https://en.wikipedia.org/wiki/Xorshift) \
[Philox](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf)