from . import core


class OxideError(RuntimeError): pass

class numeric_t: pass
class int_t(numeric_t): pass
class float_t(numeric_t): pass
class int32(int_t): pass
class float32(float_t): pass


MAXDIMS = 16;
MAXSEEDF = float(4294967295.0);


py_t = {
    int: int32,
    float: float32
}

ox_t = {
    int32: int,
    float32: float
}

tensorview_t = {
    core.TensorView_int32: int32,
    core.TensorView_float32: float32
}