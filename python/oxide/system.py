"""
SYSTEM.PY

contains module references, cpp backend (core) call mechanics
"""


from .common import *
from . import core
import atexit


backend = core.Backend()
dispatcher = core.Dispatcher(backend)

operand_module = None # reference assigned at runtime to avoid circular import


# run cpp function by name
def run(function, *args, **kwargs):
    core.mem_optimize(backend)
    try:
        result = getattr(core, function)(*args, **kwargs)
    except core.oxide_error as e:
        flush()
        throw(str(e)) # convert core.oxide_error into the equivalent python class
    except BaseException as e:
        flush()
        raise e
    
    flush()
    return result


def logw(msg): # log warning, displays immediately
    print(msg)

def flush(): # log all cached warnings from the c++ side
    print(backend.flush(False), end = "")

def throw(msg): # throw error immediately
    raise OxideError(msg)
    

@atexit.register # free backend upon program exit
def free():
    core.free_backend(backend)