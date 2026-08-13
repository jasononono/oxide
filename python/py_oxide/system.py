from .common import *
from . import core
import atexit, gc


backend = core.Backend()
dispatcher = core.Dispatcher(backend)


def run(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except core.oxide_error as e:
        throw(e.what())
    except BaseException as e:
        raise e

def throw(msg):
    raise OxideError(msg)
    

@atexit.register
def free():
    core.free_backend(backend)

# the following code causes some problems rn
# TODO: make this work

# def memory_optimize(phase, info):
#     core.memory_optimize(backend)
#     print("lol")

# gc.callbacks.append(memory_optimize)