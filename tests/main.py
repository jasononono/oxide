import sys, os

sys.path.append(os.getcwd())
from python import oxide as ox


a = ox.rand([3, 5])
b = ox.transpose(a, [0, 1])
# c = ox.reshape(b, [15, 1], ox.FORCED)

# print(a.contiguous)
# print(b.contiguous)
print(a)
print(b)

# a[0] = 3
b += a

print(a)
print(b)
# print(c)