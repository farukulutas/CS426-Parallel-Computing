#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

    *matrix = malloc((*rows) * (*cols) * sizeof(float));
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

int main() {
    // Declare dimensions for W, X, and b matrices
    int W_rows, W_cols;
    int X_rows, X_cols;
    int b_rows, b_cols;
    
    // Initialize matrices
    float *W;
    float *X;
    float *b;
    float *result;
    
    // Read matrices from files
    readMatrix("weightmatrix.txt", &W_rows, &W_cols, &W, 0);
    readMatrix("input.txt", &X_rows, &X_cols, &X, 0);
    readMatrix("bias.txt", &b_rows, &b_cols, &b, 1);  // Reading bias with adjusted function

    result = malloc(W_rows * X_cols * sizeof(float));
    
    // Perform the calculation: sigmoid(W*X + b)
    for (int i = 0; i < W_rows; i++) {
        for (int j = 0; j < X_cols; j++) {
            float sum = 0.0;
            for (int k = 0; k < W_cols; k++) {
                sum += W[i * W_cols + k] * X[k * X_cols + j];
            }
            sum += b[i];  // Access bias value directly as it's a single column vector
            result[i * X_cols + j] = sigmoid(sum);
        }
    }
    
    // Write result to result.txt
    writeMatrix("result.txt", W_rows, X_cols, result);

    // Free allocated memory
    free(W);
    free(X);
    free(b);
    free(result);
    
    return 0;
}
