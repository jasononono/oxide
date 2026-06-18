COMPILER = g++
COMPFLAG = -std=c++23

INCLUDE = -Iexternal/metal-cpp -Iexternal/nanobind/include -Isrc/include

LDFLAGS = -framework Metal \
		  -framework Foundation

SRC = src/backend.cpp src/dispatcher.cpp src/functions.cpp src/tensor.cpp src/typeutil.cpp src/oxide.cpp main.cpp
OUT = main


all: $(SRC)
	$(COMPILER) $(COMPFLAG) $(SRC) $(INCLUDE) $(LDFLAGS) -o $(OUT)
	./$(OUT)

run: $(OUT)
	./$(OUT)

clean:
	rm -f $(OUT)