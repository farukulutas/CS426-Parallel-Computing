#include <mpi.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

void quickSort(std::vector < int > & arr, int low, int high);
int partition(std::vector < int > & arr, int low, int high);
int findLocalMedian(std::vector < int > & arr);

void quickSort(std::vector < int > & arr, int low, int high) {
  if (low < high) {
    int pi = partition(arr, low, high);
    quickSort(arr, low, pi - 1);
    quickSort(arr, pi + 1, high);
  }
}

int partition(std::vector < int > & arr, int low, int high) {
  int pivot = arr[high];
  int i = (low - 1);

  for (int j = low; j < high; j++) {
    if (arr[j] < pivot) {
      i++;
      std::swap(arr[i], arr[j]);
    }
  }
  std::swap(arr[i + 1], arr[high]);
  return (i + 1);
}

int findLocalMedian(std::vector < int > & arr) {
  size_t n = arr.size() / 2;
  nth_element(arr.begin(), arr.begin() + n, arr.end());
  return arr[n];
}

int main(int argc, char * argv[]) {
  MPI_Init( & argc, & argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, & rank);
  MPI_Comm_size(MPI_COMM_WORLD, & size);

  std::vector < int > data;
  std::vector < int > local_data;
  int n_data;

  if (rank == 0) {
    std::ifstream inputFile(argv[1]);
    if (!inputFile.is_open()) {
      std::cerr << "Error opening input file!" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int value;
    while (inputFile >> value) {
      data.push_back(value);
    }
    inputFile.close();
    n_data = data.size();
  }

  MPI_Bcast( & n_data, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int local_size = n_data / size;
  local_data.resize(local_size);

  MPI_Scatter(data.data(), local_size, MPI_INT,
    local_data.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

  quickSort(local_data, 0, local_data.size() - 1);

  int local_median = findLocalMedian(local_data);
  std::vector < int > all_medians(size);
  MPI_Allgather( & local_median, 1, MPI_INT, all_medians.data(), 1, MPI_INT, MPI_COMM_WORLD);
  int pivot = findLocalMedian(all_medians);

  auto pivotPos = std::partition(local_data.begin(), local_data.end(), [pivot](int x) {
    return x < pivot;
  });
  std::vector < int > lower(local_data.begin(), pivotPos);
  std::vector < int > upper(pivotPos, local_data.end());

  bool globally_sorted = false;
  while (!globally_sorted) {
    for (int step = 0; step < std::log2(size); ++step) {
      int partner = rank ^ (1 << step);
      if (partner < size) {
        int local_size = local_data.size();
        int partner_size;
        MPI_Sendrecv( & local_size, 1, MPI_INT, partner, 0, &
          partner_size, 1, MPI_INT, partner, 0,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector < int > partner_data(partner_size);
        MPI_Sendrecv(local_data.data(), local_size, MPI_INT, partner, 1,
          partner_data.data(), partner_size, MPI_INT, partner, 1,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (rank < partner) {
          local_data.insert(local_data.end(), partner_data.begin(), partner_data.end());
          std::sort(local_data.begin(), local_data.end());
          local_data.resize(local_size);
        } else {
          partner_data.insert(partner_data.end(), local_data.begin(), local_data.end());
          std::sort(partner_data.begin(), partner_data.end());
          local_data = std::vector < int > (partner_data.begin() + partner_size, partner_data.end());
        }
      }
    }

    bool local_sorted = std::is_sorted(local_data.begin(), local_data.end());
    MPI_Allreduce( & local_sorted, & globally_sorted, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
  }

  std::ofstream localFile("output" + std::to_string(rank) + ".txt");
  for (const auto & num: local_data) {
    localFile << num << "\n";
  }
  localFile.close();

  if (rank == 0) {
    std::vector < int > sorted_data(size * local_size);
    MPI_Gather(local_data.data(), local_size, MPI_INT, sorted_data.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    std::ofstream completeFile(argv[3]);
    for (int num: sorted_data) {
      completeFile << num << "\n";
    }
    completeFile.close();
  } else {
    MPI_Gather(local_data.data(), local_size, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}
