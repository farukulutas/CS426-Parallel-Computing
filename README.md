# Bilkent CS426 Parallel Computing Projects

Welcome to the repository for the CS426 - Parallel Computing course at Bilkent University. This repository contains four detailed projects that illustrate the application of parallel computing techniques, particularly focusing on CUDA and GPU programming.

## Projects Overview

1. **Project 1 - 55 / 100:** Implementation of a single layer in a fully connected neural network. Students are required to write both a serial and a parallel program that processes two matrices and a vector of float values stored in ASCII text files. The main task is to calculate the output using a sigmoid function after matrix multiplication and bias addition. This project aims to demonstrate the efficiency of parallel computing in handling neural network computations, comparing the performance of serial and parallel implementations.
2. **Project 2 - 40 / 100:** Implementation of the Quicksort algorithm in both serial and parallel versions. The project starts with a serial implementation of Quicksort, requiring students to sort a sequence of integers provided in a text file. The parallel part of the project involves using MPI to implement Quicksort on a hypercube topology, emphasizing efficient parallel sorting and load balancing among processors. This project aims to teach students about parallelizing traditional algorithms like Quicksort and understanding the complexities involved in distributed computing environments.
3. **Project 3 - 95 / 100:** Implementation of a Convolutional Neural Network (CNN). Initially, students are tasked with developing a serial code version, followed by a parallel implementation using OpenMP. The project revolves around reading data from text files and performing operations like convolution, zero-padding, max-pooling, and applying a sigmoid activation function. The emphasis is on understanding the structure and functionality of CNNs, while also learning to optimize the code for parallel performance using OpenMP and profiling tools. This project is instrumental in teaching the application of parallel computing techniques in the field of deep learning and image processing.
4. **Project 4 - 90 / 100:** Implementation of a basic k-mer search program in both serial and parallel versions, using CUDA for GPU programming. This project involves handling substrings of a given length (k-mers) within a reference string and multiple read strings, with an emphasis on performance optimization and algorithm efficiency.

## Technologies and Techniques

- **CUDA:** Utilized for parallel computing on NVIDIA GPUs.
- **GPU Programming:** Leveraging the power of GPUs for handling intensive computational tasks.
- **Parallel Algorithms:** Implementation of efficient algorithms suitable for parallel computation.
- **MPI (Message Passing Interface):** Used for programming parallel computers. A key technology in Project 2 for implementing parallel Quicksort on a hypercube topology.
- **OpenMP (Open Multi-Processing):** An application programming interface (API) that supports multi-platform shared-memory multiprocessing programming, crucial in Project 3 for parallelizing CNN layers.
  
## Getting Started

Please refer to individual project directories for specific instructions on compiling and running each project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

Bilkent University | CS426 - Parallel Computing
