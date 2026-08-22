from .common import *
from . import core
import atexit


backend = core.Backend()
dispatcher = core.Dispatcher(backend)


def run(function, *args, **kwargs):
    core.mem_optimize(backend)
    try:
        return getattr(core, function)(*args, **kwargs)
    except core.oxide_error as e:
        throw(str(e))
    except BaseException as e:
        raise e

def throw(msg):
    raise OxideError(msg)
    

@atexit.register
def free():
    core.free_backend(backend)