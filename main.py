from python import oxide


a = oxide.random([2, 4], 1, 2, oxide.float32)
b = oxide.random(4, 1, 1, oxide.float32)

print(oxide.add(a, b))




# TODO: xorshift -> some better rand function idk
# TODO: slicing
# TODO: matmul
# TODO: delayed execution (dispatch gpu & let cpu cook on its own until results are needed again)
# TODO: refactor tensor.py
# TODO: change cpp header structure? (functions.xpp -> operand.xpp, create.xpp etc.)