import numpy as np
import torch
import py_oxide as ox
from time import perf_counter
import tracemalloc


PRECISION = 4


def speed_add(method, size):
    tracemalloc.start()
    mem_start, _ = tracemalloc.get_traced_memory()

    if method == "numpy":
        a = np.random.random_sample(size)
        b = np.random.random_sample(size)
        start = perf_counter()
        out = np.add(a, b)
    if method == "torch_cpu":
        a = torch.rand(size, device = "cpu")
        b = torch.rand(size, device = "cpu")
        start = perf_counter()
        out = torch.add(a, b)
    if method == "torch_gpu":
        a = torch.rand(size, device = "mps")
        b = torch.rand(size, device = "mps")
        torch.mps.synchronize()
        start = perf_counter()
        out = torch.add(a, b)
        torch.mps.synchronize()
    if method == "oxide":
        a = ox.rand([size])
        b = ox.rand([size])
        start = perf_counter()
        out = ox.add(a, b)
    
    mem_cur, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"method '{method}', size = {size}")
    print(f"    time = {round(perf_counter() - start, PRECISION)}s, mem_cur = {round(mem_cur / 10**6, PRECISION)}MB, mem_peak = {round(mem_peak / 10**6, PRECISION)}MB")

speed_add("numpy", 100000000)
speed_add("torch_cpu", 100000000)
speed_add("torch_gpu", 100000000)
speed_add("oxide", 100000000)