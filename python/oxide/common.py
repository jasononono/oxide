"""
COMMON.PY

misc inline functions, constants, and types (parallel to common.hpp)
"""


class OxideError(BaseException): pass

# a clearer inheritance hierarchy is used here to make type checking & python type conversion easier
class oxide_t: pass
class numeric_t(oxide_t): pass
class int_t(numeric_t): pass
class float_t(numeric_t): pass
class int32(int_t): pass
class float32(float_t): pass


MAXDIMS = 16
MAXSEEDF = float(4294967295.0)
CACHETHRESHOLD = 1024 * 16


def with_type(dtype, name):
    return name + "_" + dtype.__name__

# convert oxide data into python type
def pycast(dtype, data):
    if isinstance(OX_T[dtype], type):
        return OX_T[dtype](data)
    return OX_T[dtype][0](data)

def ansi(code, str):
    return f"\x1b[{code}m{str}\x1b[0m"


PY_T = { # corresponding oxide dtype of python types
    int: int32,
    float: float32
}
OX_T = { # corresponding python type of oxide dtypes (if a tuple is provided, dtype conversion will prefer the first element)
    int32: int,
    float32: (float, int)
}