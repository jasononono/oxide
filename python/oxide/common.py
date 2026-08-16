from . import core


class OxideError(BaseException): pass

class numeric_t: pass
class int_t(numeric_t): pass
class float_t(numeric_t): pass
class int32(int_t): pass
class float32(float_t): pass


MAXDIMS = 16;
MAXSEEDF = float(4294967295.0);


PY_T = { # corresponding oxide dtype of python types
    int: int32,
    float: float32
}
OX_T = { # corresponding python type of oxide dtypes
    int32: int,
    float32: float
}
TENSOR_T = { # corresponding oxide dtype of backend tensorview classes
    core.TensorView_int32: int32,
    core.TensorView_float32: float32
}