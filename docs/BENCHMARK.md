# Benchmark
Oxide Tensors' performance are tested against similar libraries in two different criteria: *speed* and *memory*.

**These results are specific to my device and my (likely flawed) testing program, so don't take this as the definitive performance of any of the tested libraries!**

Libraries tested against Oxide:
- Numpy, CPU
- Torch, CPU (tensors are syncronized before recording results)
- Torch, GPU (tensors are syncronized before recording results)

### Procedure
The testing script will perform element-wise addition on pre-initialized arrays/tensors.
- random values in range [0, 1), stored as 32-bit floats
- has shape [3, 3, ..., 3] (16 dimensions), i.e. 43046721 total elements.

### Results

```
method 'numpy'
    time = 0.0376s, mem_cpu = 328.4219MB
method 'torch_cpu'
    time = 0.0132s, mem_cpu = 166.9688MB
method 'torch_gpu'
    time = 2.3165s, mem_cpu = 16.0781MB
method 'oxide'
    time = 0.1055s, mem_cpu = 164.1719MB
```

### Observations
- Oxide uses less CPU memory compared to CPU based methods
- Somehow outperforms Torch GPU (likely due to Oxide using a shared buffer, making data transportation more efficient at the cost of doubling memory)