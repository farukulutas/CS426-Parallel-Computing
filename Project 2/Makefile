# Makefile for qsort.cpp and hyperqsort.cpp

# Compiler settings - Can change to match your system
CXX = g++
MPI_CXX = mpicxx

# Compiler flags
CXXFLAGS = -Wall -O2
MPI_CXXFLAGS = -Wall -O2

# Build targets
all: qsort hyperqsort

qsort: qsort.cpp
	$(CXX) $(CXXFLAGS) qsort.cpp -o qsort

hyperqsort: hyperqsort.cpp
	$(MPI_CXX) $(MPI_CXXFLAGS) hyperqsort.cpp -o hyperqsort

clean:
	rm -f qsort hyperqsort
