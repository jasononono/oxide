from . import core


class OxideError(RuntimeError): pass

class numeric_t: pass
class int_t(numeric_t): pass
class float_t(numeric_t): pass
class int32(int_t): pass
class float32(float_t): pass


pyconv_t = {
    int: int32,
    float: float32
}

tensorview_t = {
    core.TensorView_int32: int32,
    core.TensorView_float32: float32
}