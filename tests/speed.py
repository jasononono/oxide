import numpy as np
import torch
import py_oxide as ox
from time import perf_counter
import tracemalloc, math


PRECISION = 4


def speed_add(method, shape):
    tracemalloc.start()
    mem_start, _ = tracemalloc.get_traced_memory()

    if method == "numpy":
        a = np.random.random_sample(shape)
        b = np.random.random_sample(shape)
        start = perf_counter()
        out = np.add(a, b)
    if method == "torch_cpu":
        a = torch.rand(shape, device = "cpu")
        b = torch.rand(shape, device = "cpu")
        start = perf_counter()
        out = torch.add(a, b)
    if method == "torch_gpu":
        a = torch.rand(shape, device = "mps")
        b = torch.rand(shape, device = "mps")
        torch.mps.synchronize()
        start = perf_counter()
        out = torch.add(a, b)
        torch.mps.synchronize()
    if method == "oxide":
        a = ox.rand(shape)
        b = ox.rand(shape)
        start = perf_counter()
        out = ox.add(a, b)
    
    mem_cur, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"method '{method}', size = {math.prod(shape)}")
    print(f"    time = {round(perf_counter() - start, PRECISION)}s, cpu_mem_cur = {round(mem_cur / 10**6, PRECISION)}MB, cpu_mem_peak = {round(mem_peak / 10**6, PRECISION)}MB")

SHAPE = [10] * 8
speed_add("numpy", SHAPE)
speed_add("torch_cpu", SHAPE)
speed_add("torch_gpu", SHAPE)
speed_add("oxide", SHAPE)