from python import py_oxide as oxide


a = oxide.rand([2, 2])
b = oxide.rand([2, 3])
print(a)
print(b)
print(oxide.add(a, b))