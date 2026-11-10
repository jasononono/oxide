import sys, os

sys.path.append(os.getcwd())
from python import oxide as ox


a = ox.rand([3, 5])
b = ox.reshape(a, [5, 4])

print(a)
print(b)