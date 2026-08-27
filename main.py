from python import oxide


a = oxide.tensor([1.0, 3.0, 5.0])
b = oxide.zeros(3, oxide.float32)

print(a*b)




# TODO: xorshift -> some better rand function idk (research philox)
# TODO: slicing
# TODO: matmul
# TODO: delayed execution (dispatch gpu & let cpu cook on its own until results are needed again)
# TODO: refactor parse_iterable()
# TODO: change cpp header structure? (functions.xpp -> operand.xpp, create.xpp etc.)
# TODO: documentation
# TODO: journal
# TODO: precompile shader?
# TODO: view as_type? (convert to type only when needed, not the entire data)