#include "../collectives.h"

#include <vector>
#include <string>

#include "sycl/sycl.hpp"

constexpr int ALIGNMENT = 64;

void TestCollectivesGPU(std::vector<size_t>& sizes, std::vector<size_t> iterations) {
  char* env_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (env_str == nullptr) {
    throw std::runtime_error("Could not find OMPI_COMM_WORLD_LOCAL_RANK!");
  }

  int local_rank = std::stoi(std::string(env_str));
  InitCollectives(local_rank);

  int mpi_size, mpi_rank;
  if(MPI_Comm_Size(MPI_COMM_WORLD, &mpi_size) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_size failed with an error");
  }
  if(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_rank failed with an error");
  }

  auto propList = sycl::property_list{sycl::property::queue::in_order()};
  sycl::queue q(sycl::gpu_selector_v, propList);

  for (size_t i = 0; i < sizes.size(); ++i) {
    auto size = sizes[i];
    auto iters = iterations[i];

    float* cpu_data = new float[size];

    float* data = sycl::aligned_alloc_device<float>(ALIGNMENT, size, q);

    for (size_t iter = 0; iter < iters; iter++) {
      for (size_t j = 0; j < size; ++j) {
        cpu_data[j] = 1.0f;
      }

      q.memcpy(data, cpu_data, sizeof(float) * size);

      float* output;
      RingAllreduce(data, size, output);

      // check result
      q.memcpy(cpu_data, output, sizeof(float) * size).wait();
      for (size_t j = 0; j < size; ++j) {
        // TODO: comparing float
        if (cpu_data[j] != (float)mpi_size) {
          throw std::runtime_error("RingAllreduce result check failed.");
        }
      }

      sycl::free(output, q);
    }

    sycl::free(data, q);
    delete[] cpu_data;
  }
}

int main() {
  std::vector<size_t> buf_sizes = {
      0, 32, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 8388608, 67108864, 536870912
  };

  std::vector<size_t> iterations = {
      100000, 100000, 100000, 100000,
      1000, 1000, 1000, 1000,
      100, 50, 10, 1
  };

  TestCollectivesGPU(buf_sizes, iterations);

  MPI_Finalize();

  return 0;
}