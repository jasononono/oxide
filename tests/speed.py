import numpy as np
import torch
import oxide as ox
from time import perf_counter
import psutil, math, os, importlib


PRECISION = 4

process = psutil.Process(os.getpid())


def speed_add(method, shape):
    mem_start = process.memory_info().rss
    start = perf_counter()

    if method == "numpy":
        a = np.random.random_sample(shape)
        b = np.random.random_sample(shape)
        out = np.add(a, b)
    if method == "torch_cpu":
        a = torch.rand(shape, device = "cpu", dtype=torch.float32)
        b = torch.rand(shape, device = "cpu", dtype=torch.float32)
        out = torch.add(a, b)
    if method == "torch_gpu":
        a = torch.rand(shape, device = "mps", dtype=torch.float32)
        b = torch.rand(shape, device = "mps", dtype=torch.float32)
        out = torch.add(a, b)
        torch.mps.synchronize()
    if method == "oxide":
        a = ox.rand(shape)
        b = ox.rand(shape)
        out = ox.add(a, b)
    
    mem = process.memory_info().rss - mem_start
    print(f"method '{method}', size = {math.prod(shape)}")
    print(f"    time = {round(perf_counter() - start, PRECISION)}s, mem = {round(mem/1024/1024, PRECISION)}MB")

SHAPE = [3] * 16
speed_add("numpy", SHAPE)
speed_add("torch_cpu", SHAPE)
speed_add("torch_gpu", SHAPE)
speed_add("oxide", SHAPE)