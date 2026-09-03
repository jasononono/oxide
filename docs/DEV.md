# how oxide works

I tried to group all functions created and used in Oxide into distinct **layers of abstraction**:

    - L0: shader program (direct GPU communication)
    - L1: backend: handles, random generators, memory management, etc.
    - L2: dispatcher: object that uses backend to configure the GPU and perform arbitrary computations
    - L3: functions & tensors: provides a usable c++ interface and perform basic checks (e.g. no negative dimensions)
    - L4: py-oxide: python interface, performs more checks (types, python-specific implementation)



# data type specialize pattern




// yap about #defines & specialize pattern
things that need to be considered when addingh a new datatype
random gen thing
namin convention for _int32
commons in hpp (3 places)

https://numpy.org/devdocs/user/basics.broadcasting.html
https://numpy.org/devdocs/user/basics.copies.html