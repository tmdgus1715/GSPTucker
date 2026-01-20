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

#ifndef OPTIMIZER_HPP_
#define OPTIMIZER_HPP_

#include <vector>

#include "common/human_readable.hpp"
#include "gsptucker/constants.hpp"
#include "gsptucker/tensor.hpp"

namespace supertensor {
namespace gsptucker {

#define OPTIMIZER_TEMPLATE template <typename TensorType>
#define OPTIMIZER_TEMPLATE_ARGS TensorType

/**
 * @brief Optimizer class for Tucker decomposition.
 *
 * This class defines the optimizer for the Tucker decomposition program. The
 * optimizer is responsible for determining the partition type for the input
 * tensor, partitioning the tensor based on non-zero elements or dimensions, and
 * calculating the number of non-zero elements per task. It also computes the
 * data size for each CUDA execution sequence, ensuring efficient resource
 * utilization.
 *
 * @tparam TensorType The type of tensor used in the decomposition.
 *
 * @details The optimizer plays a crucial role in dividing the tensor into
 * manageable blocks for parallel processing on GPUs, based on either
 * dimension-based or non-zero-based partitioning strategies.
 *
 * @version 1.0.0
 * @date 2021-08-10
 *
 */
OPTIMIZER_TEMPLATE
class Optimizer {
  using tensor_t = TensorType;  ///< Type alias for the tensor type.
  using index_t =
      typename tensor_t::index_t;  ///< Type alias for the index type.
  using value_t =
      typename tensor_t::value_t;  ///< Type alias for the value type.

 public:
  /**
   * @brief Default constructor for the Optimizer.
   */
  Optimizer() {}

  /**
   * @brief Destructor for the Optimizer.
   */
  ~Optimizer() {}

  /**
   * @brief Initializes the optimizer with the necessary parameters.
   *
   * @param new_gpu_count The number of GPUs available for computation.
   * @param new_rank The Tucker rank.
   * @param new_gpu_mem_size The memory size available on each GPU.
   * @param new_data Pointer to the tensor data to be optimized.
   * @param new_cuda_stream_count The number of CUDA streams.
   */
  void Initialize(unsigned short new_gpu_count, unsigned int new_rank,
                  uint64_t new_gpu_mem_size, tensor_t* new_data,
                  int new_cuda_stream_count = 1);

  /**
   * @brief Calculates the total size of the tensor data.
   *
   * @return The total data size in bytes.
   */
  size_t GetAllDataSize();

  /**
   * @brief Calculates the total size of data to be transferred between memory
   * and GPUs.
   *
   * @return The total transfer size in bytes.
   */
  size_t GetAllTransferSize();

  /**
   * @brief Determines the optimal partition parameters.
   *
   * @return A pointer to the partition dimensions.
   */
  index_t* FindPartitionParms();

  std::vector<uint64_t> FindSkewedBlocks(
      const std::vector<uint64_t>& block_nnz_counts);

  /**
   * @brief Converts the optimizer's state to a string representation.
   *
   * This function is useful for debugging or logging the state of the
   * optimizer.
   */
  void ToString();

 private:
  /**
   * @brief Determines the partition type for the input tensor.
   *
   * This method selects whether to use dimension-based or non-zero-based
   * partitioning based on the characteristics of the input tensor.
   */
  void _DeterminePartitionType();

  /**
   * @brief Partitions the tensor based on the number of non-zero elements.
   */
  void _NonzeroBasedPartitioning();

  /**
   * @brief Partitions the tensor based on its dimensions.
   */
  void _DimensionBasedPartitioning();

  /**
   * @brief Determines the next axis or dimension for partitioning.
   *
   * @return The index of the next axis to be partitioned.
   */
  unsigned short _NextPartitionAxis();

  /**
   * @brief Adjusts the block dimensions using partition dimensions.
   */
  void _RefreshBlockDims();

  /**
   * @brief Calculates the available number of non-zero elements per task.
   */
  void _AvailableNonzeroCountPerTask();

  /* Methods to get data sizes for various CUDA execution sequences */

  /**
   * @brief Gets the data size for the input tensor.
   *
   * @return The data size in bytes.
   */
  size_t _get_data_size_input_tensor();

  /**
   * @brief Gets the data size for the core tensor.
   *
   * @return The data size in bytes.
   */
  size_t _get_data_size_core_tensor();

  /**
   * @brief Gets the data size for all factor matrices.
   *
   * @return The data size in bytes.
   */
  size_t _get_data_size_all_factors();

  /**
   * @brief Gets the data size for the delta values.
   *
   * @return The data size in bytes.
   */
  size_t _get_data_size_delta();

  /* Methods to get data sizes for sub-tensors and intermediate results */

  /**
   * @brief Gets the data size for a sub-tensor.
   *
   * @return The data size in bytes.
   */
  size_t _get_data_size_sub_tensor();

  /**
   * @brief Gets the data size for sub-factor matrices.
   *
   * @return The data size in bytes.
   */
  size_t _get_data_size_sub_factors();

  /**
   * @brief Gets the data size for sub-delta values.
   *
   * @return The data size in bytes.
   */
  size_t _get_data_size_sub_delta();

  /**
   * @brief Gets the data transfer size for a sub-tensor.
   *
   * @return The transfer size in bytes.
   */
  size_t _get_transfer_size_sub_tensor();

  /**
   * @brief Gets the data transfer size for the core tensor.
   *
   * @return The transfer size in bytes.
   */
  size_t _get_transfer_size_core_tensor();

  /**
   * @brief Gets the data transfer size for sub-factor matrices.
   *
   * @return The transfer size in bytes.
   */
  size_t _get_transfer_size_sub_factors();

  /**
   * @brief Gets the data transfer size for delta values.
   *
   * @return The transfer size in bytes.
   */
  size_t _get_transfer_size_delta();

 public:
  int cuda_stream_count;     ///< The number of CUDA streams.
  unsigned int rank;         ///< The Tucker rank.
  unsigned short gpu_count;  ///< The number of GPUs available.

  index_t* block_dims;  ///< The dimensions of each block in the tensor.
  index_t*
      partition_dims;    ///< The dimensions used for partitioning the tensor.
  uint64_t block_count;  ///< The total number of blocks in the tensor.

  uint64_t avg_nnz_count_per_block;   ///< The estimated number of non-zero
                                      ///< elements per task.
  uint64_t avail_nnz_count_per_task;  ///< The available number of non-zero
                                      ///< elements per task.

  gsptucker::enums::PartitionTypes
      partition_type;      ///< The type of partitioning used.
  size_t gpu_mem_size;     ///< The size of GPU memory available.
  bool use_avg_partition;  ///< Flag to enable average-based partitioning.

 private:
  tensor_t* _data;  ///< Pointer to the tensor data.
};
}  // namespace gsptucker
}  // namespace supertensor

#include "gsptucker/optimizer.tpp"
#endif /* OPTIMIZER_HPP_ */