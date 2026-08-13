from python import py_oxide as oxide


a = oxide.tensor([1.0, 2.0, 3.0])
b = oxide.tensor([2, 3, 4])

o = oxide.add(a, b)
print(o)