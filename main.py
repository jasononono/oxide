from python import py_oxide as oxide


a = oxide.tensor([1, 2, 3])
b = oxide.tensor([2, 3, 4])
print(a, b)
print(oxide.add(a, b))