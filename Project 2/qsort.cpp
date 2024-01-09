#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iterator>

void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

int partition (std::vector<int> &arr, int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quickSort(std::vector<int> &arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

void printArray(const std::vector<int> &arr, std::ofstream &output) {
    for (int i : arr) {
        output << i << '\n';
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <inputfile> <outputfile>\n";
        return 1;
    }

    std::string inputFileName = argv[1];
    std::string outputFileName = argv[2];

    std::ifstream inputFile(inputFileName);
    std::ofstream outputFile(outputFileName);

    if (!inputFile.is_open() || !outputFile.is_open()) {
        std::cerr << "Error opening files!" << std::endl;
        return 1;
    }

    std::vector<int> arr;
    int value;
    while (inputFile >> value) {
        arr.push_back(value);
    }

    quickSort(arr, 0, arr.size() - 1);
    printArray(arr, outputFile);

    inputFile.close();
    outputFile.close();

    return 0;
}
