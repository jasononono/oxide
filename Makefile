COMPILER = g++
COMPFLAG = -std=c++23

INCLUDE = -Imetal-cpp

LDFLAGS = -framework Metal \
		  -framework Foundation

SRC = src/backend.cpp src/tensor.cpp src/dispatcher.cpp main.cpp
OUT = main


all: $(SRC)
	$(COMPILER) $(COMPFLAG) $(SRC) $(INCLUDE) $(LDFLAGS) -o $(OUT)
	./$(OUT)

run: $(OUT)
	./$(OUT)

clean:
	rm -f $(OUT)