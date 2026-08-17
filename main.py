from python import oxide


def f():
    a = oxide.zeros([256*16])

f()
oxide.core.mem_optimize(oxide.system.backend)
print(oxide.system.backend.mem_cacheget())