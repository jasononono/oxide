from python import oxide


a = oxide.zeros([2, 4])
b = oxide.transpose(a, [1, 0])

print(b)