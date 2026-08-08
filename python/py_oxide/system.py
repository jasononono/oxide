from . import core
import atexit, gc


backend = core.Backend()
dispatcher = core.Dispatcher(backend)


def call(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except RuntimeError:
        

@atexit.register
def free():
    core.free_backend(backend)

# the following code causes some problems rn
# TODO: make this work

# def memory_optimize(phase, info):
#     core.memory_optimize(backend)
#     print("lol")

# gc.callbacks.append(memory_optimize)