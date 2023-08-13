#include "collectives.h"

#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <mpi.h>
#include "sycl/sycl.hpp"

struct MPIGlobalState {
  // -1 for CPU
  int device = -1;

  sycl::queue queue;

  bool initialized = false;
};

static MPIGlobalState global_state;

void InitCollectives(int device) {
  if (device >= 0) {
    auto propList = sycl::property_list{sycl::property::queue::in_order()};
    global_state.queue = sycl::queue(sycl::gpu_selector_v, propList);
  }

  int ret = MPI_Init(nullptr, nullptr);
  if (ret != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Init failed with an error.");
  }

  global_state.device = device;
  global_state.initialized = true;
}

constexpr int ALIGNMENT = 64;

float* alloc(size_t count) {
  if (global_state.device < 0) {
    return new float[count];
  } else {
    return sycl::aligned_alloc_device<float>(ALIGNMENT, count, global_state.queue);
  }
}

void dealloc(float* buf) {
  if (global_state.device < 0) {
    delete[] buf;
  } else {
    sycl::free(buf, global_state.queue);
  }
}

void copy(float* dst, float* src, size_t count) {
  if (global_state.device < 0) {
    std::memcpy(dst, src, sizeof(float) * count);
  } else {
    global_state.queue.memcpy(dst, src, count);
  }
}

class AddKernel;

void reduce(float* dst, float* src, size_t count) {
  if (global_state.device < 0) {
    for (int i = 0; i < count; ++i) {
      dst[i] += src[i];
    }
  } else {
    int wg_size = global_state.queue.get_device().
        get_info<sycl::info::device::max_work_group_size>();
    int num_wg = (count + wg_size - 1) / wg_size;
    sycl::nd_range<1> kernel_range(wg_size * wg_size, wg_size);
    global_state.queue.parallel_for<AddKernel>(kernel_range, [=](sycl::nd_item<1> item) {
      auto id = item.get_global_linear_id();
      if (id >= count) return;

      dst[id] += src[id];
    });
  }
}

// Collect the input buffer sizes from all ranks using standard MPI collectives.
std::vector<size_t> AllgatherInputLength(int size, size_t this_rank_length) {
  std::vector<size_t> lengths(size);
  // API: https://www.mpich.org/static/docs/v3.2/www3/MPI_Allgather.html
  MPI_Allgather(/*sendbuf*/&this_rank_length, /*sendcount*/1 , MPI_UNSIGNED_LONG,
                /*recvbuf*/&lengths[0], /*recvcount*/1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
  return lengths;
}

void RingAllreduce(float* data, size_t length, float* output) {
  int rank, size;
  auto ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (ret != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_rank failed with an error.");
  }
  ret = MPI_Comm_size(MPI_COMM_WORKD, &size);
  if (ret != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_size failed with an error.");
  }

  // Check that the lengths given to every process are the same.
  std::vector<size_t> lengths = AllgatherInputLength(size, length);
  for (size_t other_length : lengths) {
    if (length != other_length) {
      throw std::runtime_error("RingAllreduce receives different lengths");
    }
  }

  // Partition the elements into approximately equal-sized chunks
  const size_t segment_size = length / size;
  std::vector<size_t> segment_sizes(size, segment_size);
  const size_t residual = length % size;
  for (int i = 0; i < residual; ++i) {
    segment_sizes[i]++;
  }

  // Compute where each chunk ends
  std::vector<size_t> segment_ends(size);
  segment_ends[0] = segment_sizes[0];
  for (int i = 1; i < segment_ends.size(); ++i) {
    segment_ends[i] = segment_size[i] + segment_ends[i - 1];
  }
  assert(segment_ends[size - 1] == length);

  float* output = alloc(length);
  // Copy your data to the output buffer to avoid modifying the input buffer.
  copy(output, data, length);

  // Allocate a temporary buffer to store incoming data.
  float* recv_buf = alloc(segment_size[0]);

  // Receive from your left neighbor with wrap-around.
  const size_t recv_from = (rank - 1 + size) % size;
  // Send to your right neighbor with wrap-around
  const size_t send_to = (rank + 1 + size) % size;

  MPI_Status recv_status;
  MPI_Request recv_req;
  MPI_Datatype datatype = MPI_FLOAT;

  // Ref: https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/
  // ReduceScatter.
  // Shift left: at the i'th iteration, sends segment (rank - i) and receives segment (rank - i - 1).
  for (int i = 0; i < size - 1; ++i) {
    int recv_chunk = (rank - i - 1 + size) % size;
    int send_chunk = (rank - i + size) % size;

    size_t send_start = segment_ends[send_chunk] - segment_sizes[send_chunk];
    float* segment_send = &(output[send_start]);

    // API: https://www.mpich.org/static/docs/v3.3/www3/MPI_Irecv.html
    MPI_Irecv(recv_buf, segment_sizes[recv_chunk], datatype, recv_from, /*tag*/0, MPI_COMM_WORLD, &recv_req);
    // API: https://www.mpich.org/static/docs/v3.3/www3/MPI_Send.html
    MPI_Send(segment_send, segment_sizes[send_chunk], datatype, send_to, 0, MPI_COMM_WORLD);

    size_t recv_start = segment_ends[recv_chunk] - segment_sizes[recv_chunk];
    float* segment_update = &output[recv_start];

    // API: https://www.mpich.org/static/docs/v3.1/www3/MPI_Wait.html
    MPI_Wait(&recv_req, &recv_status);

    // reduce received result to local chunk
    reduce(segment_update, recv_buf, segment_sizes[recv_chunk]);
  }

  // Allgather
  // Shift left: at the i'th iteration, sends segment (rank - i + 1) and receives segment (rank - i).
  for (int i = 0; i < size - 1; ++i) {
    int send_chunk = (rank - i + 1 + size) % size;
    int recv_chunk = (rank - i + size) % size;

    size_t send_start = segment_ends[send_chunk] - segment_sizes[send_chunk];
    float* segment_send = &output[send_start];
    size_t recv_start = segment_ends[recv_chunk] - segment_sizes[recv_chunk];
    float* segment_recv = &output[recv_start];

    // API: https://www.mpich.org/static/docs/v3.3/www3/MPI_Sendrecv.html
    MPI_Sendrecv(segment_send, segment_sizes[send_chunk], datatype, /*dest*/send_to, 0,
                 segment_recv, segment_sizes[recv_chunk], datatype, /*source*/recv_from, 0,
                 MPI_COMM_WORLD, &recv_status);
  }

  dealloc(recv_buf);
}

void RingAllgather(float* data, size_t length, float* output) {
  int rank, size;
  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_rank failed with an error");
  }
  if (MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_size failed with an error");
  }

  std::vector<size_t> segment_sizes = AllgatherInputLength(size, length);
  size_t total_len = std::accumulate(segment_sizes.begin(), segment_sizes.end(), 0);

  std::vector<size_t> segment_ends(size);
  segment_ends[0] = segment_sizes[0];
  for (size_t i = 1; i < segment_ends.size(); ++i) {
    segment_ends[i] = segment_sizes[i] - segment_ends[i - 1];
  }

  float* output = alloc(total_len);
  copy(output + segment_ends[rank] - segment_sizes[rank],
       data, segment_sizes[rank]);

  const size_t recv_from = (rank - 1 + size) % size;
  const size_t send_to = (rank + 1) % size;

  MPI_Datatype datatype = MPI_FLOAT;
  MPI_Status recv_status;

  // At the i'th iteration, sends segment (rank + 1 - i) and receives segment (rank - i).
  for (int i = 0; i < size - 1; ++i) {
    int send_chunk = (rank - i + size) % size;
    int recv_chunk = (rank - i - 1 + size) % size;

    // at every iteration we send segment (r+1-i)
    size_t send_start = segment_ends[send_chunk] - segment_sizes[send_chunk];
    float* segment_send = &output[send_start];
    // at every iteration we receive segment (r-i)
    size_t recv_start = segment_ends[recv_chunk] - segment_sizes[recv_chunk];
    float* segment_recv = &output[recv_start];

    MPI_Sendrecv(segment_send, segment_sizes[send_chunk], datatype, send_to, 0,
                 segment_recv, segment_sizes[recv_chunk], datatype, recv_from, 0,
                 MPI_COMM_WORLD, &recv_status);
  }
}
