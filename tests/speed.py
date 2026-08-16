import numpy as np
import torch
import oxide as ox
from time import perf_counter
import psutil, math, os, random, gc, time, io


PRECISION = 4

process = psutil.Process(os.getpid())
buf = io.StringIO()


@torch.compiler.disable()
def speed_add(method, shape):

    axes = list(range(len(shape)))
    random.shuffle(axes)

    if method == "numpy":

        a = np.random.random_sample(shape)#.transpose(axes)
        b = np.random.random_sample(shape)#.transpose(axes)

        start = perf_counter()
        mem_start = process.memory_info().rss
        out = np.add(a, b)
        print(a[*([0] * len(shape))], file = buf)

    if method == "torch_cpu":

        a = torch.rand(shape, device = "cpu", dtype=torch.float32)#.permute(axes)
        b = torch.rand(shape, device = "cpu", dtype=torch.float32)#.permute(axes)
       
        start = perf_counter()
        mem_start = process.memory_info().rss
        out = torch.add(a, b)
        torch.cpu.synchronize()
        print(a[*([0] * len(shape))], file = buf)

    if method == "torch_gpu":

        a = torch.rand(shape, device = "mps", dtype=torch.float32)#.permute(axes)
        b = torch.rand(shape, device = "mps", dtype=torch.float32)#.permute(axes)

        start = perf_counter()
        mem_start = process.memory_info().rss
        out = torch.add(a, b)
        torch.mps.synchronize()
        print(a[*([0] * len(shape))], file = buf)

    if method == "oxide":

        # a = ox.transpose(ox.rand(shape), axes)
        # b = ox.transpose(ox.rand(shape), axes)
        a = ox.rand(shape)
        b = ox.rand(shape)
        
        start = perf_counter()
        mem_start = process.memory_info().rss
        out = ox.add(a, b)
        print(a[*([0] * len(shape))], file = buf)
    
    mem = process.memory_info().rss - mem_start
    print(f"method '{method}', size = {math.prod(shape)}")
    print(f"    time = {round(perf_counter() - start, PRECISION)}s, mem_cpu = {round(mem/1024/1024, PRECISION)}MB")

    gc.collect()
    time.sleep(3)

SHAPE = [3] * 16
speed_add("numpy", SHAPE)
speed_add("torch_cpu", SHAPE)
speed_add("torch_gpu", SHAPE)
speed_add("oxide", SHAPE)

buf.close()