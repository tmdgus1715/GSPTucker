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

#ifndef CUDA_AGENT_HPP_
#define CUDA_AGENT_HPP_

#include "cuda_runtime_api.h"

#include "common/memory_region.hpp"
#include "gsptucker/tensor.hpp"

namespace supertensor {
namespace gsptucker {

/**
 * @brief CUDAAGENT_TEMPLATE
 * @details This macro defines the template for the CUDA agent class.
 */
#define CUDAAGENT_TEMPLATE template <typename TensorType>
#define CUDAAGENT_TEMPLATE_ARGS TensorType

/**
 * @brief CUDA agent class for managing CUDA resources in Tucker decomposition.
 *
 * The `CudaAgent` class is responsible for managing CUDA devices, streams, and memory
 * regions in the Tucker decomposition program. It handles the allocation of device buffers,
 * creation of CUDA streams, and manages interactions with the GPU during tensor operations.
 *
 * @tparam TensorType The type of the tensor to be managed by the CUDA agent.
 *
 * @details The class provides methods to allocate device buffers, set up CUDA streams,
 * and manage memory regions required for efficient GPU computations. It also provides
 * access to device properties and manages multiple CUDA streams for concurrent operations.
 *
 * @author Jihye Lee
 * @date 2023-08-10
 * @version 1.0.0
 */
CUDAAGENT_TEMPLATE
class CudaAgent {
public:
  using tensor_t = TensorType;                 ///< Type alias for the tensor type.
  using index_t = typename tensor_t::index_t;  ///< Type alias for the tensor's index type.
  using value_t = typename tensor_t::value_t;  ///< Type alias for the tensor's value type.
  using memrgn_t = common::MemoryRegion<char>; ///< Type alias for the memory region type.

  /**
   * @brief Structure to hold device buffers for CUDA operations.
   *
   * This structure contains memory regions for indices, values, and addresses of the
   * tensor and core data required for Tucker decomposition. It also includes memory
   * regions for factor matrices and delta values used during computations.
   */
  struct DeviceBuffer {
    using int_mr_t = common::MemoryRegion<int>;               ///< Memory region for integer data.
    using index_mr_t = common::MemoryRegion<index_t>;         ///< Memory region for index data.
    using value_mr_t = common::MemoryRegion<value_t>;         ///< Memory region for value data.
    using addr_mr_t = common::MemoryRegion<std::uintptr_t *>; ///< Memory region for address pointers.

    index_mr_t X_indices[gsptucker::constants::kMaxOrder]; ///< Indices of the tensor.
    value_mr_t X_values;                                   ///< Values of the tensor.
    addr_mr_t X_idx_addr;                                  ///< Addresses of the tensor indices.

    index_mr_t core_indices[gsptucker::constants::kMaxOrder]; ///< Indices of the core tensor.
    value_mr_t core_values;                                   ///< Values of the core tensor.
    addr_mr_t core_idx_addr;                                  ///< Addresses of the core tensor indices.

    value_mr_t factors[gsptucker::constants::kMaxOrder]; ///< Factor matrices.
    addr_mr_t factor_addr;                               ///< Addresses of the factor matrices.

    value_mr_t delta; ///< Delta values used in updates.
  };

  /**
   * @brief Default constructor.
   *
   * Initializes a `CudaAgent` object with default settings.
   */
  CudaAgent();

  /**
   * @brief Constructor with device ID.
   *
   * Initializes a `CudaAgent` object with a specific CUDA device ID.
   *
   * @param id The CUDA device ID to be used.
   */
  CudaAgent(unsigned id);

  /**
   * @brief Destructor.
   *
   * Cleans up allocated resources and CUDA streams.
   */
  ~CudaAgent();

  /**
   * @brief Allocates the maximum buffer size required for CUDA operations.
   *
   * @return `true` if the allocation is successful, `false` otherwise.
   */
  bool AllocateMaximumBuffer();

  /**
   * @brief Creates a specified number of CUDA streams.
   *
   * @param stream_count The number of streams to create.
   * @return `true` if the streams are created successfully, `false` otherwise.
   */
  bool CreateCudaStream(unsigned int stream_count);

  /**
   * @brief Sets up device buffers for tensor operations.
   *
   * This function initializes the device buffers with the tensor data, the Tucker rank,
   * and the maximum number of non-zero elements in any block.
   *
   * @param tensor Pointer to the tensor to be processed.
   * @param rank The Tucker rank.
   * @param max_nnz_count_in_block The maximum number of non-zero elements in a block.
   */
  void SetDeviceBuffers(tensor_t *tensor, int rank, uint64_t max_nnz_count_in_block);

  /**
   * @brief Converts the CUDA agent's state to a string representation.
   *
   * This function generates a string that represents the current state of the `CudaAgent`,
   * including its device ID, stream count, and allocated memory size.
   *
   * @note Useful for debugging or logging the agent's state.
   */
  void ToString();

  /* Getter methods */

  /**
   * @brief Retrieves the CUDA device ID used by the agent.
   *
   * @return The CUDA device ID.
   */
  int get_device_id() const { return this->_device_id; }

  /**
   * @brief Retrieves the CUDA streams managed by the agent.
   *
   * @return Pointer to the array of CUDA streams.
   */
  cudaStream_t *get_cuda_streams() { return this->_streams; }

  /**
   * @brief Retrieves the properties of the CUDA device.
   *
   * @return Pointer to the CUDA device properties structure.
   */
  cudaDeviceProp *get_device_properties() { return &this->_device_properties; }

  /**
   * @brief Retrieves the base memory region used by the agent.
   *
   * @return Pointer to the base memory region.
   */
  memrgn_t *get_base_memory_region() { return this->_base_mr; }

  /**
   * @brief Retrieves the number of CUDA streams created by the agent.
   *
   * @return The number of CUDA streams.
   */
  int get_stream_count() const { return this->_stream_count; }

  /**
   * @brief Retrieves the number of CUDA devices available.
   *
   * @return The number of CUDA devices.
   */
  int get_device_count() const { return this->_device_count; }

  /**
   * @brief Retrieves the total size of memory allocated by the agent.
   *
   * @return The size of allocated memory in bytes.
   */
  size_t get_allocated_size() const { return this->_allocated_size; }

  /* Setter methods */

  /**
   * @brief Sets the number of CUDA streams to be created.
   *
   * @param stream_count The number of streams to create.
   */
  void set_stream_count(unsigned int stream_count) { this->_stream_count = stream_count; }

public:
  DeviceBuffer dev_buf; ///< Device buffer structure containing CUDA memory regions.

private:
  int _device_id;                    ///< CUDA device ID.
  unsigned _stream_count;            ///< Number of CUDA streams.
  cudaStream_t *_streams;            ///< Array of CUDA streams.
  cudaDeviceProp _device_properties; ///< CUDA device properties.
  memrgn_t *_base_mr;                ///< Base memory region.
  int _device_count;                 ///< Number of CUDA devices.
  size_t _allocated_size;            ///< Total allocated memory size.

}; // class CudaAgent

} // namespace gsptucker
} // namespace supertensor

#include "gsptucker/cuda_agent.tpp"

#endif // CUDA_AGENT_HPP_
