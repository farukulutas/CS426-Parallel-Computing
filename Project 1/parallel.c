#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Function prototypes
void readMatrix(const char *filename, int *rows, int *cols, float **matrix, int isBias);
void writeMatrix(const char *filename, int rows, int cols, float *matrix);
float sigmoid(float x);

// Function to read a matrix from a file
void readMatrix(const char *filename, int *rows, int *cols, float **matrix, int isBias) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open file %s.\n", filename);
        exit(1);
    }

    fscanf(file, "%d", rows);
    if (isBias) {
        *cols = 1;  // For bias vector, set columns to 1
    } else {
        fscanf(file, "%d", cols);
    }

    *matrix = (float *)malloc((*rows) * (*cols) * sizeof(float));
    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            fscanf(file, "%f", &(*matrix)[i * (*cols) + j]);
        }
    }

    fclose(file);
}

// Function to write a matrix to a file
void writeMatrix(const char *filename, int rows, int cols, float *matrix) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Could not open file %s.\n", filename);
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%.5f ", matrix[i * cols + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

// Sigmoid function
float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int W_rows, W_cols, X_rows, X_cols, b_rows, b_cols;
    float *W = NULL, *X = NULL, *b = NULL, *result = NULL, *local_result = NULL, *local_W = NULL, *local_b = NULL;
    int rows_per_process, local_rows;

    // Master process
    if (world_rank == 0) {
        // Reading matrices
        readMatrix("weightmatrix.txt", &W_rows, &W_cols, &W, 0);
        readMatrix("input.txt", &X_rows, &X_cols, &X, 0);
        readMatrix("bias.txt", &b_rows, &b_cols, &b, 1);

        // Compute rows to be processed by each process
        rows_per_process = W_rows / world_size;
        result = malloc(W_rows * X_cols * sizeof(float));
    }

    // Broadcast the matrix dimensions using point-to-point communication
    for (int i = 0; i < world_size; i++) {
        if (i == world_rank) continue;
        if (world_rank == 0) {
            MPI_Send(&W_rows, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&W_cols, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&X_rows, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
            MPI_Send(&X_cols, 1, MPI_INT, i, 3, MPI_COMM_WORLD);
            MPI_Send(&b_rows, 1, MPI_INT, i, 4, MPI_COMM_WORLD);
        } else {
            MPI_Recv(&W_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&W_cols, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&X_rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&X_cols, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&b_rows, 1, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Calculate local rows for each process
    local_rows = (world_rank == world_size - 1) ? (W_rows - rows_per_process * (world_size - 1)) : rows_per_process;

    // Allocate memory for local matrices
    local_result = malloc(local_rows * X_cols * sizeof(float));
    local_W = (world_rank == 0) ? W : malloc(local_rows * W_cols * sizeof(float));
    local_b = (world_rank == 0) ? b : malloc(local_rows * sizeof(float));

    // Distribute W and b to all processes
    if (world_rank == 0) {
        for (int i = 1; i < world_size; ++i) {
            int start_row = i * rows_per_process;
            int num_rows = (i == world_size - 1) ? (W_rows - rows_per_process * i) : rows_per_process;
            MPI_Send(&W[start_row * W_cols], num_rows * W_cols, MPI_FLOAT, i, 5, MPI_COMM_WORLD);
            MPI_Send(&b[start_row], num_rows, MPI_FLOAT, i, 6, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(local_W, local_rows * W_cols, MPI_FLOAT, 0, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_b, local_rows, MPI_FLOAT, 0, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Broadcast X to all processes
    if (world_rank != 0) {
        X = malloc(X_rows * X_cols * sizeof(float));
    }
    for (int i = 0; i < world_size; i++) {
        if (i == world_rank) continue;
        if (world_rank == 0) {
            MPI_Send(X, X_rows * X_cols, MPI_FLOAT, i, 7, MPI_COMM_WORLD);
        } else {
            MPI_Recv(X, X_rows * X_cols, MPI_FLOAT, 0, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Local computation
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < X_cols; j++) {
            float sum = 0.0;
            for (int k = 0; k < W_cols; k++) {
                sum += local_W[i * W_cols + k] * X[k * X_cols + j];
            }
            sum += local_b[i];
            local_result[i * X_cols + j] = sigmoid(sum);
        }
    }

    // Gather the results from all processes to the master process
    if (world_rank == 0) {
        memcpy(result, local_result, local_rows * X_cols * sizeof(float));
        for (int i = 1; i < world_size; ++i) {
            int start_row = i * rows_per_process;
            int num_rows = (i == world_size - 1) ? (W_rows - rows_per_process * i) : rows_per_process;
            MPI_Recv(&result[start_row * X_cols], num_rows * X_cols, MPI_FLOAT, i, 8, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_result, local_rows * X_cols, MPI_FLOAT, 0, 8, MPI_COMM_WORLD);
    }

    // Writing result to file and freeing memory
    if (world_rank == 0) {
        writeMatrix("result.txt", W_rows, X_cols, result);
        free(result);
    }
    if (world_rank != 0 || local_W != W) {
        free(local_W);
    }
    if (world_rank != 0 || local_b != b) {
        free(local_b);
    }
    if (world_rank != 0) {
        free(X);
    }
    free(local_result);

    MPI_Finalize();
    return 0;
}
