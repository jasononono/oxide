from python import oxide


for i in range(60):
    a = oxide.zeros([2, 4])
    b = oxide.transpose(a, [1, 0])
print(b)

oxide.core.memory_optimize(oxide.system.backend)
print(oxide.system.backend.mem_cacheget())