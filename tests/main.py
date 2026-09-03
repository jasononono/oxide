import sys, os

sys.path.append(os.getcwd())
from python import oxide as ox


a = ox.random([3, 3, 3], 0, 9, ox.int32)

print(ox.topy(a))
print(a)