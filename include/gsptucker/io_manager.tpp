/*
 * This file is part of GSPTucker.
 *
 * GSPTucker is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GSPTucker is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GSPTucker.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <omp.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "common/human_readable.hpp"
#include "gsptucker/helper.hpp"
#include "gsptucker/io_manager.hpp"

namespace supertensor {
namespace gsptucker {

IO_MANAGER_TEMPLATE
IOManager<IO_MANAGER_ARGS>::IOManager() {}

IO_MANAGER_TEMPLATE
IOManager<IO_MANAGER_ARGS>::IOManager(const std::string& input_path,
                                      const std::string& output_path)
    : _input_path(input_path) {
  if (output_path.empty()) {
    _ssd_paths.push_back("~");
  } else {
    _ssd_paths.push_back(output_path);
  }

  printf("Multi-SSD I/O Configuration (default paths):\n");
  printf("  Requested SSDs: 1\n");
  printf("  SSD[0]: %s (blocks: %s/blocks, output: %s/output)\n",
         _ssd_paths[0].c_str(), _ssd_paths[0].c_str(), _ssd_paths[0].c_str());
}

IO_MANAGER_TEMPLATE
std::string IOManager<IO_MANAGER_ARGS>::_GetSSDPath(uint64_t block_id) const {
  // Round-robin distribution across SSDs
  return _ssd_paths[0];
}

IO_MANAGER_TEMPLATE
bool IOManager<IO_MANAGER_ARGS>::ParseFromFile(TensorType** tensor) {
  std::ifstream file(this->_input_path);
  if (!file.is_open()) {
    std::string err_msg =
        "[ERROR] Canot open file \"" + this->_input_path + "\" for reading...";
    throw std::runtime_error(ERROR_LOG(err_msg));
    return false;
  }
  file.seekg(0, file.end);

  size_t file_size = static_cast<size_t>(file.tellg());
  assert(file_size > 0);
  std::cout << "Input Tensor Size (COO) \t: "
            << common::HumanReadable{(std::uintmax_t)file_size} << std::endl;

  file.seekg(0, file.beg);
  std::string buffer(file_size, '\0');
  file.read(&buffer[0], file_size);
  file.close();

  return _ReadData(buffer.c_str(), file_size, tensor);
}

IO_MANAGER_TEMPLATE
bool IOManager<IO_MANAGER_ARGS>::ParseMetadataFromFile(tensor_t** tensor) {
  std::ifstream file(this->_input_path);
  if (!file.is_open()) {
    std::string err_msg = "[ERROR] Cannot open file \"" + this->_input_path +
                          "\" for reading metadata...";
    throw std::runtime_error(ERROR_LOG(err_msg));
    return false;
  }

  // Get file size for progress tracking
  file.seekg(0, file.end);
  size_t file_size = static_cast<size_t>(file.tellg());
  assert(file_size > 0);
  std::cout << "Input Tensor Size (COO) \t: "
            << common::HumanReadable{(std::uintmax_t)file_size} << std::endl;
  file.seekg(0, file.beg);

  return this->_ReadMetadataStream(file, tensor, file_size);
}

IO_MANAGER_TEMPLATE
bool IOManager<IO_MANAGER_ARGS>::_ReadData(const char* buffer,
                                           const size_t buffer_length,
                                           tensor_t** tensor) {
  int thread_id = 0;
  int thread_count = 0;
  int order = (*tensor)->order;

  std::vector<uint64_t>* pos;
  std::vector<index_t>* local_max_dims;
  std::vector<index_t>* local_dim_offset;
  std::vector<uint64_t> nnz_prefix_sum;

  index_t* global_max_dims;
  uint64_t global_nnz_count = 0;

  value_t* values;
  index_t* indices[order];

#pragma omp parallel private(thread_id)
  {
    thread_id = omp_get_thread_num();
    thread_count = omp_get_num_threads();

// Initialize local variables
#pragma omp single
    {
      pos = new std::vector<uint64_t>[thread_count];
      local_max_dims = new std::vector<index_t>[thread_count];
      local_dim_offset = new std::vector<index_t>[thread_count];
      nnz_prefix_sum.resize(thread_count);
    }
    pos[thread_id].push_back(0);
    local_max_dims[thread_id].resize(order);
    local_dim_offset[thread_id].resize(order);

    for (unsigned short axis = 0; axis < order; ++axis) {
      local_max_dims[thread_id][axis] = std::numeric_limits<index_t>::min();
      local_dim_offset[thread_id][axis] = std::numeric_limits<index_t>::max();
    }
    // 1. Find '\n' : the number of nonzeros
#pragma omp for reduction(+ : global_nnz_count)
    for (size_t sz = 0; sz < buffer_length; ++sz) {
      if (buffer[sz] == '\n') {
        global_nnz_count++;
        pos[thread_id].push_back(sz + 1);
      }
    }

#pragma omp barrier
    if (thread_id > 0) {
      pos[thread_id].front() = pos[thread_id - 1].back();
    }
#pragma omp barrier
#pragma omp single
    {
      // prefix sum
      nnz_prefix_sum[0] = 0;
      for (int tid = 1; tid < thread_count; ++tid) {
        nnz_prefix_sum[tid] =
            nnz_prefix_sum[tid - 1] + (pos[tid - 1].size() - 1);
      }
      assert(nnz_prefix_sum.back() + pos[thread_count - 1].size() - 1 ==
             global_nnz_count);

      global_max_dims = gsptucker::allocate<index_t>(order);
      for (unsigned short axis = 0; axis < order; ++axis) {
        global_max_dims[axis] = std::numeric_limits<index_t>::min();
        indices[axis] = gsptucker::allocate<index_t>(global_nnz_count);
      }
      values = gsptucker::allocate<value_t>(global_nnz_count);
    }
    uint64_t nnz_count_tid = pos[thread_id].size();
    for (uint64_t nnz = 1; nnz < nnz_count_tid; ++nnz) {
      // Calculate the starting position of the current slice in the buffer
      const int len = pos[thread_id][nnz] - pos[thread_id][nnz - 1] - 1;
      uint64_t buff_ptr = pos[thread_id][nnz - 1];
      char* buff = const_cast<char*>(&buffer[buff_ptr]);

      // Tokenize	the slice by newline characters
      char* rest = strtok_r(buff, "\n", &buff);
      char* token;

      if (rest != NULL) {
        // Calculate the offset of the current thread in the global index
        uint64_t offset = nnz_prefix_sum[thread_id];
        unsigned short axis = 0;
        /* Coordinate */
        // Loop through each coordinate in the slice
        while ((token = strtok_r(rest, " \t", &rest)) && (axis < order)) {
          index_t idx = strtoull(token, NULL, 10);

          // Update the maximum and minimum indices for the current axis
          local_max_dims[thread_id][axis] =
              std::max<index_t>(local_max_dims[thread_id][axis], idx);
          local_dim_offset[thread_id][axis] =
              std::min<index_t>(local_dim_offset[thread_id][axis], idx);

          // Store the current index in the global indices array,
          // with 1-indexing (subtract 1 from idx)
          indices[axis][offset + nnz - 1] = idx - 1;  // 1-Indexing
          ++axis;
        }  // !while

        /* Value */
        // Parse the value of the current slice
        value_t val;
        if (token != NULL) {
          val = std::stod(token);
        } else {
          // If the slice does not have a value, generate a random one between 0
          // and 1
          val = gsptucker::frand<value_t>(0, 1);
        }
        values[offset + nnz - 1] = val;
      }
    }  // !for

// 2. extract metadata for the tensor (dims, offsets, and #nnzs)
#pragma omp critical
    {
      // Update the global max dimension for the axis
      for (unsigned short axis = 0; axis < order; ++axis) {
        global_max_dims[axis] = std::max<index_t>(
            global_max_dims[axis], local_max_dims[thread_id][axis]);
        if (local_dim_offset[thread_id][axis] < 1) {
          // outputs are based on base-0 indexing
          throw std::runtime_error(ERROR_LOG(
              "We note that input tensors must follow base-1 indexing"));
        }
      }
    }  // !omp critical
  }  //! omp

  uint64_t block_id = 0;

  (*tensor)->set_dims(global_max_dims);
  (*tensor)->set_nnz_count(global_nnz_count);

  (*tensor)->MakeBlocks(1, &global_nnz_count);
  (*tensor)->InsertData(block_id, &indices[0], values);
  (*tensor)->blocks[block_id]->ToString();

  // Deallocate
  delete[] pos;
  delete[] local_max_dims;
  delete[] local_dim_offset;
  nnz_prefix_sum.clear();
  std::vector<uint64_t>().swap(nnz_prefix_sum);

  return true;
}

IO_MANAGER_TEMPLATE
bool IOManager<IO_MANAGER_ARGS>::_ReadMetadataStream(std::ifstream& file,
                                                     tensor_t** tensor,
                                                     size_t file_size) {
  const int order = (*tensor)->order;
  const size_t LINES_PER_BATCH =
      10000000;  // Process 10,000,000 lines at a time

  std::vector<index_t> global_max_dims(order,
                                       std::numeric_limits<index_t>::min());
  uint64_t global_nnz_count = 0;

  std::vector<std::string> line_batch;
  line_batch.reserve(LINES_PER_BATCH);
  std::string line;

  while (file.good()) {
    // Read a batch of lines
    line_batch.clear();
    for (size_t i = 0; i < LINES_PER_BATCH && std::getline(file, line); ++i) {
      if (!line.empty()) {
        line_batch.push_back(line);
      }
    }

    if (line_batch.empty()) break;

    const size_t num_lines = line_batch.size();

    // Parallel processing of lines in this batch
    std::vector<std::vector<index_t>> local_max_dims(
        omp_get_max_threads(),
        std::vector<index_t>(order, std::numeric_limits<index_t>::min()));
    std::vector<uint64_t> local_nnz_counts(omp_get_max_threads(), 0);

#pragma omp parallel
    {
      const int thread_id = omp_get_thread_num();

#pragma omp for
      for (size_t line_idx = 0; line_idx < num_lines; ++line_idx) {
        const std::string& current_line = line_batch[line_idx];

        local_nnz_counts[thread_id]++;

        // Parse coordinates to find max dimensions
        std::istringstream iss(current_line);
        std::string token;
        unsigned short axis = 0;

        while (axis < order && iss >> token) {
          index_t idx = strtoull(token.c_str(), NULL, 10);
          if (idx < 1) {
            throw std::runtime_error(ERROR_LOG(
                "We note that input tensors must follow base-1 indexing"));
          }
          local_max_dims[thread_id][axis] =
              std::max<index_t>(local_max_dims[thread_id][axis], idx);
          ++axis;
        }
      }
    }  // end omp parallel

    // Merge results from this batch
    for (int tid = 0; tid < omp_get_max_threads(); ++tid) {
      global_nnz_count += local_nnz_counts[tid];
      for (unsigned short axis = 0; axis < order; ++axis) {
        global_max_dims[axis] =
            std::max<index_t>(global_max_dims[axis], local_max_dims[tid][axis]);
      }
    }
  }

  file.close();

  // Set tensor metadata (dimensions and nnz count only)
  index_t* dims = gsptucker::allocate<index_t>(order);
  for (unsigned short axis = 0; axis < order; ++axis) {
    dims[axis] = global_max_dims[axis];
  }

  (*tensor)->set_dims(dims);
  (*tensor)->set_nnz_count(global_nnz_count);

  std::cout << "  Lines per batch: " << LINES_PER_BATCH << std::endl;
  std::cout << "  NNZ count: " << global_nnz_count << std::endl;
  std::cout << "  Dimensions: ";
  for (unsigned short axis = 0; axis < order; ++axis) {
    std::cout << dims[axis];
    if (axis < order - 1) std::cout << " x ";
  }
  std::cout << std::endl;

  return true;
}

IO_MANAGER_TEMPLATE
template <typename OptimizerType>
void IOManager<IO_MANAGER_ARGS>::CreateTensorBlocks(TensorType** src,
                                                    TensorType** dest,
                                                    OptimizerType* optimizer) {
  using tensor_t = TensorType;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;

  printf("... 1) Creating tensor blocks\n");
  const unsigned short order = (*src)->order;
  const index_t* const dims = (*src)->dims;
  const uint64_t nnz_count = (*src)->nnz_count;

  const index_t* const block_dims = optimizer->block_dims;
  const index_t* const partition_dims = optimizer->partition_dims;
  const uint64_t block_count = optimizer->block_count;

  index_t** indices = (*src)->blocks[0]->indices;
  value_t* values = (*src)->blocks[0]->values;

  // 1. Count nonzeros per block
  std::vector<std::vector<uint64_t>> local_nnz_histograms(
      omp_get_max_threads(), std::vector<uint64_t>(block_count, 0));
  std::vector<std::vector<index_t>> local_nnz_coords(
      omp_get_max_threads(), std::vector<index_t>(order, 0));

  printf("... 2) Counting nonzeros per block\n");
#pragma omp parallel
  {
    const int thread_id = omp_get_thread_num();
    const int thread_count = omp_get_num_threads();

#pragma omp for
    for (uint64_t nnz = 0; nnz < nnz_count; ++nnz) {
      // Convert coordinates of nonzero into block id
      uint64_t block_id = 0;
      uint64_t mult = 1;
      for (unsigned short iter = 0; iter < order; ++iter) {
        unsigned short axis = order - iter - 1;

        assert(indices[axis][nnz] < dims[axis] &&
               "Coordinate is out of bounds");
        index_t block_idx = indices[axis][nnz] / block_dims[axis];
        assert(block_idx < partition_dims[axis] &&
               "Block coordinate is out of bounds");
        block_id += block_idx * mult;
        mult *= partition_dims[axis];
      }
      assert(block_id < block_count);
      ++local_nnz_histograms[thread_id][block_id];
    }  // !omp for
  }  // omp parallel

  printf("... 3) Creating blocks\n");
  uint64_t check_nnz_count = 0;
  std::vector<uint64_t> global_nnz_histogram(block_count, 0);
  for (int tid = 0; tid < omp_get_max_threads(); ++tid) {
    for (uint64_t block_id = 0; block_id < block_count; ++block_id) {
      global_nnz_histogram[block_id] += local_nnz_histograms[tid][block_id];
      check_nnz_count += local_nnz_histograms[tid][block_id];
    }
  }

  assert(check_nnz_count == nnz_count);
  (*dest)->set_partition_dims(partition_dims);
  (*dest)->MakeBlocks(block_count, global_nnz_histogram.data());

  printf("... 4) Inserting data\n");
  value_t NormX = 0.0f;
  omp_lock_t lck;
  omp_init_lock(&lck);

  for (uint64_t nnz = 0; nnz < nnz_count; ++nnz) {
    std::vector<index_t> local_tensor_coord(order, 0);
    value_t val = values[nnz];
    uint64_t block_id = 0;
    uint64_t mult = 1;
    for (unsigned short iter = 0; iter < order; ++iter) {
      unsigned short axis = order - iter - 1;

      assert(indices[axis][nnz] < dims[axis] && "Coordinate is out of bounds");
      local_tensor_coord[axis] = indices[axis][nnz];
      index_t block_idx = indices[axis][nnz] / block_dims[axis];

      assert(block_idx < partition_dims[axis] &&
             "Block coordinate is out of bounds");
      block_id += block_idx * mult;
      mult *= partition_dims[axis];
    }
    assert(block_id < block_count);

    // Direct insertion into block, no child logic
    uint64_t pos = global_nnz_histogram[block_id];
    --global_nnz_histogram[block_id];
    (*dest)->blocks[block_id]->InsertNonzero(pos, local_tensor_coord.data(),
                                             val);

    NormX += val * val;
  }

  // Compute norm
  (*dest)->norm = std::sqrt(NormX);

  delete *src;
  *src = nullptr;

  printf("Assign indices in blocks\n");
  assert(block_count != 0);

#pragma omp parallel for
  for (uint64_t block_id = 0; block_id < block_count; ++block_id) {
    (*dest)->blocks[block_id]->AssignIndicesToEachMode();

    this->WriteRecordToStorage(block_id, (*dest));
    (*dest)->blocks[block_id]->DeallocateRecord();
  }

  global_nnz_histogram.clear();
  std::vector<uint64_t>().swap(global_nnz_histogram);
  printf("... 5) Done\n");
}

IO_MANAGER_TEMPLATE
void IOManager<IO_MANAGER_ARGS>::WriteRecordToStorage(uint64_t block_id,
                                                      tensor_t* tensor) {
  std::string ssd_path = _GetSSDPath(block_id);
  std::string output_dir = ssd_path + "/output";
  std::string mkdir_cmd = "mkdir -p " + output_dir;
  system(mkdir_cmd.c_str());

  block_t* block = tensor->blocks[block_id];

  std::string record_path =
      output_dir + "/record_" + std::to_string(block_id) + ".bin";
  std::ofstream record_file(record_path, std::ios::trunc | std::ios::binary);
  if (!record_file.is_open()) {
    std::string err_msg =
        "[ERROR] Cannot open file \"" + record_path + "\" for writing...";
    throw std::runtime_error(ERROR_LOG(err_msg));
  }

  const uint64_t record_size = tensor->blocks[block_id]->GetRecordSize();
  char* record_buffer = gsptucker::allocate<char>(record_size);

  try {
    uint64_t record_size = tensor->blocks[block_id]->Serialize(record_buffer);
    if (record_size > record_size) {
      throw std::runtime_error(ERROR_LOG(
          "[ERROR] Record size is larger than the fixed record size."));
    }
    record_file.write(record_buffer, record_size);
  } catch (...) {
    record_file.close();
    gsptucker::deallocate(record_buffer);
    throw;
  }
  record_file.close();
  gsptucker::deallocate(record_buffer);
}

IO_MANAGER_TEMPLATE
void IOManager<IO_MANAGER_ARGS>::ReadRecordFromStorage(uint64_t chunk_id,
                                                       tensor_t* tensor,
                                                       char* record_buffer) {
  std::string ssd_path = _GetSSDPath(chunk_id);
  std::string record_path =
      ssd_path + "/output/record_" + std::to_string(chunk_id) + ".bin";
  std::ifstream record_file(record_path, std::ios::binary);
  if (!record_file.is_open()) {
    std::string err_msg =
        "[ERROR] Cannot open file \"" + record_path + "\" for reading...";
    throw std::runtime_error(ERROR_LOG(err_msg));
  }

  const uint64_t record_size = tensor->blocks[chunk_id]->GetRecordSize();

  try {
    record_file.read(record_buffer, record_size);
    tensor->blocks[chunk_id]->Deserialize(record_buffer);
  } catch (...) {
    record_file.close();
    throw;
  }
  record_file.close();
}

}  // namespace gsptucker
}  // namespace supertensor
