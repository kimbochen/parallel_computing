# Boost library directories
HEADER_DIR=/usr/local/include/
LIB_DIR=/usr/local/lib

# Library flags
BOOST= -I$(HEADER_DIR) -L$(LIB_DIR)
OMP=-Xpreprocessor -fopenmp -lomp

# All arguments
CXX=g++-9
CXXFLAGS=-std=c++17 -O3 -Wall -Werror $(OMP) $(BOOST)
TARGET = hw1

.PHONY: clean

all: hw1

hw1: $(TARGET).cc
	$(CXX) $^ $(CXXFLAGS) -o $@

run: hw1
	./hw1 samples/01.txt

clean:
	rm -rf $(TARGET)

