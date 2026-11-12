import sys, os

sys.path.append(os.getcwd())
from python import oxide as ox


a = ox.rand([3, 5])

print((a[0][2] + 123).value())