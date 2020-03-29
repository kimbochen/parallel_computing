CXX=g++
CXXFLAGS=-std=c++17 -O3 -fopenmp -ltbb -Wall -Werror
TARGET=hw1

.PHONY: clean

all: hw1

hw1: $(TARGET).cc
	$(CXX) $^ $(CXXFLAGS) -o $@

run: hw1
	 srun -c4 ./hw1 samples/04.txt

clean:
	rm -rf hw1
