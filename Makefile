CXX=g++-9

HEADER_DIR=/usr/local/include/
LIB_DIR=/usr/local/lib

OMP=-Xpreprocessor -fopenmp -lomp
CXXFLAGS=-std=c++17 -I$(HEADER_DIR) -L$(LIB_DIR) $(OMP) -Wall -Werror
TARGET = hw1

.PHONY: clean

all: hw1

hw1: $(TARGET).cc
	$(CXX) $^ $(CXXFLAGS) -O3 -o $@

run: hw1
	./hw1 samples/01.txt

test: $(TARGET).cc
	$(CXX) $^ $(CXXFLAGS) -O0 -DDEBUG -o $@

clean:
	rm -rf $(TARGET) test

