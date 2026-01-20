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

#include <cassert>

#include "gsptucker/constants.hpp"
#include "gsptucker/helper.hpp"
#include "gsptucker/optimizer.hpp"

namespace supertensor {
namespace gsptucker {
/**
 * @brief Initialize the optimizer
 * @details Initialize the optimizer with the given parameters
 * @param new_gpu_count Number of GPUs
 * @param new_rank Tucker rank
 * @param new_gpu_mem_size GPU memory size
 * @param new_data Tensor data
 * @param new_cuda_stream_count Number of CUDA streams
 *
 */
OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::Initialize(
    unsigned short new_gpu_count, unsigned int new_rank,
    uint64_t new_gpu_mem_size, tensor_t* new_data,
    int new_cuda_stream_count) {
  cuda_stream_count = new_cuda_stream_count;
  rank = new_rank;
  gpu_count = new_gpu_count;
  gpu_mem_size = new_gpu_mem_size;

  partition_type = gsptucker::enums::PartitionTypes::kDimensionPartition;
  this->_data = new_data;
}
/**
 * @brief Get all data size
 * @details Get all data size for a CUDA execution sequence
 * @return Sum of all data sizes
 */
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::GetAllDataSize() {
  size_t ret_size = 0;

  ret_size += this->_get_data_size_sub_tensor();
  ret_size += this->_get_data_size_core_tensor();
  ret_size += this->_get_data_size_sub_factors();
  ret_size += this->_get_data_size_sub_delta();

  return ret_size;
}
/**
 * @brief Get all transfer size
 * @details Get all amount of transfer data size for a CUDA execution sequence
 * @return Sum of all transfer data sizes
 *
 */
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::GetAllTransferSize() {
  size_t ret_size = 0;

  ret_size += this->_get_transfer_size_sub_tensor();
  ret_size += this->_get_transfer_size_core_tensor();
  ret_size += this->_get_transfer_size_sub_factors();
  ret_size += this->_get_transfer_size_delta();
  return ret_size;
}

/**
 * @brief Find optimal partition parameters
 * @details Find optimal partition parameters for the input tensor
 * @return Partition parameters
 *
 */
OPTIMIZER_TEMPLATE
Optimizer<OPTIMIZER_TEMPLATE_ARGS>::index_t*
Optimizer<OPTIMIZER_TEMPLATE_ARGS>::FindPartitionParms() {
  MYPRINT("Find partition find_partition_parms\n");

  unsigned short order = this->_data->order;
  index_t* dims = this->_data->dims;

  block_dims = gsptucker::allocate<index_t>(order);
  partition_dims = gsptucker::allocate<index_t>(order);

  for (unsigned short axis = 0; axis < order; ++axis) {
    block_dims[axis] = dims[axis];
    partition_dims[axis] = 1;
  }
  this->_RefreshBlockDims();
  // Determine partition type (Nonzero-based Partitioning OR Dimension-based
  // Partitioning)
  this->_DeterminePartitionType();

  return partition_dims;
}

OPTIMIZER_TEMPLATE
std::vector<uint64_t> Optimizer<OPTIMIZER_TEMPLATE_ARGS>::FindSkewedBlocks(
    const std::vector<uint64_t>& block_nnz_counts) {
  std::vector<uint64_t> result;
  if (block_nnz_counts.empty()) return result;

  // Compute average nnz across non-empty blocks
  uint64_t sum = 0, cnt = 0;
  for (auto v : block_nnz_counts) {
    if (v > 0) {
      sum += v;
      ++cnt;
    }
  }
  if (cnt == 0) return result;

  double thresh;
  if (this->avail_nnz_count_per_task > 0) {
    thresh = static_cast<double>(this->avail_nnz_count_per_task);
  } else {
    double skew_factor = 1.5;  // Default skew factor
    double avg = static_cast<double>(sum) / static_cast<double>(cnt);
    thresh = avg * skew_factor;
  }

  for (uint64_t id = 0; id < block_nnz_counts.size(); ++id) {
    auto nnz = block_nnz_counts[id];
    if (nnz > static_cast<uint64_t>(thresh)) {
      result.push_back(id);
    }
  }
  return result;
}

/**
 * @brief Determine partition type
 * @details Determine partition type (Nonzero-based Partitioning OR
 * Dimension-based Partitioning)
 */
OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_DeterminePartitionType() {
  MYPRINT(
      "Determine partition type (Nonzero-based Partitioning OR Dimension-based "
      "Partitioning)\n");

  size_t gpu_total_mem_size = gpu_count * gpu_mem_size;

  size_t required_size = 0;
  required_size += this->_get_data_size_input_tensor();
  required_size += gpu_count * this->_get_data_size_core_tensor();
  required_size += gpu_count * this->_get_data_size_all_factors();
  required_size += this->_get_data_size_delta();

  printf("Required Size / Total %d GPUs Mem size\t: ", gpu_count);
  std::cout << common::HumanReadable{(std::uintmax_t)required_size} << " / "
            << common::HumanReadable{(std::uintmax_t)gpu_total_mem_size}
            << std::endl;

  // if (required_size <= gpu_total_mem_size) {
  //   partition_type = gsptucker::enums::PartitionTypes::kNonzeroPartition;
  //   std::cout << "\t- Partitioning Type: Nonzero-based (Small-scale)\n"
  //             << std::endl;
  //   this->_NonzeroBasedPartitioning();
  // } else {
  partition_type = gsptucker::enums::PartitionTypes::kDimensionPartition;
  std::cout << "\t- Partitioning Type: Dimension-based (Large-scale)\n"
            << std::endl;
  this->_DimensionBasedPartitioning();
  // }
  this->_AvailableNonzeroCountPerTask();
}
/**
 * @brief Next partition axis
 * @details Determine the next axis or dimension along which the data will be
 * partitioned
 *
 */
OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_NonzeroBasedPartitioning() {
  // cuda_stream_count = 1;
}
/**
 * @brief Next partition axis
 * @details Determine the next axis or dimension along which the data will be
 * partitioned
 *
 */
OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_DimensionBasedPartitioning() {
  size_t gpu_stream_buffer_size = gpu_mem_size / cuda_stream_count;
  uint32_t total_cuda_stream_count = gpu_count * cuda_stream_count;

  int iter = 0;
  do {
    ++iter;

    unsigned short partition_axis = this->_NextPartitionAxis();
    partition_dims[partition_axis]++;
    this->_RefreshBlockDims();
  } while (!(GetAllDataSize() < gpu_stream_buffer_size &&
             block_count >= cuda_stream_count));

  // partition_dims[0] = 4;
  // partition_dims[1] = 10;
  // partition_dims[2] = 2;
  // partition_dims[3] = 1;

  this->_RefreshBlockDims();
}
/**
 * @brief Calculate available nonzeros per task
 * @details Calculate the maximum number of non-zero elements per task
 *
 */
OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_AvailableNonzeroCountPerTask() {
  size_t gpu_stream_buffer_size = gpu_mem_size / cuda_stream_count;
  size_t avail_buffer_size =
      gpu_stream_buffer_size -
      (this->_get_data_size_core_tensor() + this->_get_data_size_sub_factors());
  avail_nnz_count_per_task =
      avail_buffer_size / (this->_data->order * sizeof(index_t) +
                           sizeof(value_t) + rank * sizeof(value_t));
}

/**
 * @brief Print the optimizer information
 * @details Print the optimizer information, including the partition type,
 * partition dimensions, block count, and the number of non-zero elements per
 * task
 *
 */
OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::ToString() {
  PrintLine();
  printf("< OPTIMIZER >\n");

  std::cout << "Partition Type: ";
  if (partition_type == gsptucker::enums::kNonzeroPartition) {
    std::cout << "Nonzero-based Partitioning" << std::endl;
  } else {
    std::cout << "Dimension-based Partitioning" << std::endl;
  }

  unsigned short order = this->_data->order;
  for (int axis = 0; axis < order; ++axis) {
    printf("Partition dim[%d] = %lu\n", axis, partition_dims[axis]);
  }
  std::cout << "\t@ All data size for a CUDA execution seq.\t"
            << common::HumanReadable{(std::uintmax_t)GetAllDataSize()}
            << std::endl;

  std::cout << "\t@ All amount of transfer data size \t"
            << common::HumanReadable{(std::uintmax_t)GetAllTransferSize()}
            << std::endl;
  printf("The number of blocks: %lu\n", block_count);
  printf("Max. Available nonzeros per task: %lu\n", avail_nnz_count_per_task);
  printf("The number of CUDA Streams in a GPU: %d\n", cuda_stream_count);
  printf("The number of GPUs: %d\n", gpu_count);
}
/**
 * @brief Get data size for the input tensor
 * @details Get data size for the input tensor for a CUDA execution sequence
 * @return Data size for the input tensor
 */
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_input_tensor() {
  size_t ret_size = this->_data->nnz_count *
                    (this->_data->order * sizeof(index_t) + sizeof(value_t));
  return ret_size;
}

/**
 * @brief Get data size for a sub-tensor
 * @details Get data size for a sub-tensor for a CUDA execution sequence
 * @return Data size for a sub-tensor
 *
 */
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_sub_tensor() {
  unsigned short order = this->_data->order;
  size_t ret_size =
      avg_nnz_count_per_block * (order * sizeof(index_t) + sizeof(value_t));

  return ret_size;
}

/**
 * @brief Calculate the size of a core tensor
 * @details Calculate the size of a core tensor for a CUDA execution sequence
 * @return Size of a core tensor
 */
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_core_tensor() {
  unsigned int order = this->_data->order;
  size_t core_nnz_count = std::pow(rank, order);
  size_t ret_size =
      core_nnz_count * (order * sizeof(index_t) + sizeof(value_t));
  return ret_size;
}
/**
 * @brief Calculate the size of all factor matrices
 * @details Calculate the size of all factor matrices for a CUDA execution
 * sequence
 * @return Size of all factor matrices
 *
 */
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_all_factors() {
  size_t ret_size = 0;
  for (unsigned short axis = 0; axis < this->_data->order; ++axis) {
    ret_size += this->_data->dims[axis] * rank;
  }
  return ret_size * sizeof(value_t);
}

/**
 * @brief Calculate the size of sub-factor matrices
 * @details Calculate the size of sub-factor matrices for a CUDA execution
 * sequence
 * @return Size of sub-factor matrices
 */
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_sub_factors() {
  // sum of each sub-factor for the factor
  size_t ret_size = 0;
  for (unsigned short axis = 0; axis < this->_data->order; ++axis) {
    ret_size += block_dims[axis] * rank;
  }
  return ret_size * sizeof(value_t);
}
/**
 * @brief Calculate the size of delta
 * @details Calculate the size of delta for a CUDA execution sequence
 * @return Size of delta
 *
 */
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_delta() {
  size_t ret_size = this->_data->nnz_count * rank * sizeof(value_t);
  return ret_size;
}

/**
 * @brief Calculate the size of sub-delta
 * @details Calculate the size of sub-delta for a CUDA execution sequence
 * @return Size of sub-delta
 *
 */
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_data_size_sub_delta() {
  unsigned short order = this->_data->order;
  size_t ret_size =
      this->avg_nnz_count_per_block * this->rank * sizeof(value_t);
  return ret_size;
}
/**
 * @brief Calculate the size of transfer data for a sub-tensor
 * @details Calculate the size of transfer data for a sub-tensor for a CUDA
 * execution sequence
 * @return Size of transfer data for a sub-tensor
 *
 */
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_transfer_size_sub_tensor() {
  size_t ret_size = this->_data->nnz_count *
                    (this->_data->order * sizeof(index_t) + sizeof(value_t));
  return ret_size;
}

/**
 * @brief Calculate the size of transfer data for a core tensor
 * @details Calculate the size of transfer data for a core tensor for a CUDA
 * execution sequence
 * @return Size of transfer data for a core tensor
 */
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_transfer_size_core_tensor() {
  return this->_get_data_size_core_tensor();
}

/**
 * @brief Calculate the size of transfer data for sub-factor matrices
 * @details Calculate the size of transfer data for sub-factor matrices for a
 * CUDA execution sequence
 * @return Size of transfer data for sub-factor matrices
 */
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_transfer_size_sub_factors() {
  unsigned short order = this->_data->order;
  size_t ret_size = 0;
  for (unsigned short axis = 0; axis < order; ++axis) {
    ret_size += block_count * this->block_dims[axis] * this->rank;
  }

  ret_size *= sizeof(value_t);
  return ret_size;
}

/**
 * @brief Calculate the size of transfer data for delta
 * @details Calculate the size of transfer data for delta for a CUDA execution
 * sequence
 * @return Size of transfer data for delta
 */
OPTIMIZER_TEMPLATE
size_t Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_get_transfer_size_delta() {
  size_t ret_size = this->_data->nnz_count * this->rank * sizeof(value_t);
  return ret_size;
}

/**
 * @brief Next partition axis
 * @details Determine the next axis or dimension along which the data will be
 * partitioned
 * @return Next partition axis
 */
OPTIMIZER_TEMPLATE
unsigned short Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_NextPartitionAxis() {
  unsigned short max_axis = 0;

  for (unsigned short axis = 1; axis < this->_data->order; ++axis) {
    if (block_dims[max_axis] < block_dims[axis]) {
      max_axis = axis;
    }
  }
  return max_axis;
}

/**
 * @brief Adjust block dimensions
 * @details Adjust block dimensions using partition dimensions
 *
 */
OPTIMIZER_TEMPLATE
void Optimizer<OPTIMIZER_TEMPLATE_ARGS>::_RefreshBlockDims() {
  // Initialize block dimensions
  int order = this->_data->order;
  index_t* dims = this->_data->dims;

  block_count = 1;
  for (unsigned short axis = 0; axis < order; ++axis) {
    block_dims[axis] =
        (dims[axis] + partition_dims[axis] - 1) / partition_dims[axis];
    index_t check_dim = (dims[axis] + block_dims[axis] - 1) / block_dims[axis];
    if (check_dim != partition_dims[axis]) {
      throw std::runtime_error(ERROR_LOG(
          "[ERROR] Block dimension is larger than the tensor dimension."));
    }
    block_count *= partition_dims[axis];
  }

  avg_nnz_count_per_block =
      (this->_data->nnz_count + block_count - 1) / block_count;
}
}  // namespace gsptucker
}  // namespace supertensor
