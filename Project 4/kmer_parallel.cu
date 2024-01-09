#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_REF_LENGTH 1000000
#define MAX_READ_LENGTH 200
#define MAX_READS 20480

__global__ void countKmers(char *ref, char *reads, int *counts, int *readLengths, int refLen, int numReads, int k, int totalKmers) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= totalKmers) return;

    int readIdx = 0, kmerIdx = tid;
    while (kmerIdx >= readLengths[readIdx] - k + 1) {
        kmerIdx -= readLengths[readIdx] - k + 1;
        readIdx++;
    }

    // Dynamically allocate kmer
    char *kmer = new char[k];
    for (int i = 0; i < k; i++) {
        kmer[i] = reads[readIdx * MAX_READ_LENGTH + kmerIdx + i];
    }

    // Count occurrences in reference
    int count = 0;
    for (int i = 0; i <= refLen - k; i++) {
        bool match = true;
        for (int j = 0; j < k; j++) {
            if (ref[i + j] != kmer[j]) {
                match = false;
                break;
            }
        }
        if (match) count++;
    }
    counts[tid] = count;

    // Free dynamically allocated memory
    delete[] kmer;
}

int main(int argc, char *argv[]) {
    if(argc != 5) {
        printf("Wrong arguments usage: ./kmer_serial [REFERENCE_FILE] [READ_FILE] [k] [OUTPUT_FILE]\n" );
        return 1;
    }

    char *refFile = argv[1];
    char *readsFile = argv[2];
    int k = atoi(argv[3]);
    char *outputFile = argv[4];

    // Read reference sequence
    char ref[MAX_REF_LENGTH] = {0};
    FILE *fp = fopen(refFile, "r");
    if (!fp) {
        printf("Error: File not found %s\n", refFile);
        return 1;
    }
    fgets(ref, MAX_REF_LENGTH, fp);
    fclose(fp);
    int refLen = strlen(ref);

    // Read reads
    char reads[MAX_READS * MAX_READ_LENGTH] = {0};
    int readLengths[MAX_READS] = {0};
    fp = fopen(readsFile, "r");
    if (!fp) {
        printf("Error: File not found %s\n", readsFile);
        return 1;
    }
    char read[MAX_READ_LENGTH];
    int numReads = 0;
    while (fgets(read, MAX_READ_LENGTH, fp) && numReads < MAX_READS) {
        int readLen = strlen(read);
        if (read[readLen - 1] == '\n') read[--readLen] = '\0';
        strcpy(&reads[numReads * MAX_READ_LENGTH], read);
        readLengths[numReads] = readLen;
        numReads++;
    }
    fclose(fp);

    // Calculate totalKmers
    int totalKmers = 0;
    for (int i = 0; i < numReads; i++) {
        totalKmers += readLengths[i] > k ? readLengths[i] - k + 1: 0;
    }

    // Allocate memory on GPU
    char *d_ref, *d_reads;
    int *d_counts, *d_readLengths;
    if (cudaMalloc((void **)&d_ref, refLen * sizeof(char)) != cudaSuccess ||
        cudaMalloc((void **)&d_reads, numReads * MAX_READ_LENGTH * sizeof(char)) != cudaSuccess ||
        cudaMalloc((void **)&d_counts, totalKmers * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void **)&d_readLengths, numReads * sizeof(int)) != cudaSuccess) {
        printf("Error: cudaMalloc failed\n");
        return 1;
    }

    // Copy data to GPU
    cudaMemcpy(d_ref, ref, refLen * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reads, reads, numReads * MAX_READ_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_readLengths, readLengths, numReads * sizeof(int), cudaMemcpyHostToDevice);

    // Run kernel
    int threadsPerBlock = 256;
    int blocks = (totalKmers + threadsPerBlock - 1) / threadsPerBlock;
    countKmers<<<blocks, threadsPerBlock>>>(d_ref, d_reads, d_counts, d_readLengths, refLen, numReads, k, totalKmers);
    cudaDeviceSynchronize();

    // Retrieve results
    int *counts = (int *)malloc(totalKmers * sizeof(int));
    cudaMemcpy(counts, d_counts, totalKmers * sizeof(int), cudaMemcpyDeviceToHost);

    // Write results to file
    fp = fopen(outputFile, "w");
    for (int i = 0, idx = 0; i < numReads; i++) {
        int totalReadCount = 0;
        for (int j = 0; j < readLengths[i] - k + 1; j++, idx++) {
            totalReadCount += counts[idx];
        }
        fprintf(fp, "%d\n", totalReadCount);
    }
    fclose(fp);

    // Free memory
    cudaFree(d_ref);
    cudaFree(d_reads);
    cudaFree(d_counts);
    cudaFree(d_readLengths);
    free(counts);

    return 0;
}
