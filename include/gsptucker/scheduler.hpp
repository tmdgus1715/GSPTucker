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

#ifndef SCHEDULER_HPP_
#define SCHEDULER_HPP_

#include <vector>
#include <atomic>
#include <condition_variable>
#include <mutex>

#include "common/circlequeue.hpp"
#include "gsptucker/tensor.hpp"

namespace supertensor {
namespace gsptucker {

template <typename TaskType>
struct QueueCtx {
  CircularQueue<TaskType> q;
  std::mutex mtx;
  std::condition_variable cv_has_task;
  std::condition_variable cv_has_space;
  std::atomic<uint64_t> pending_nnz{0};
  std::atomic<bool> production_done{false};
  explicit QueueCtx(size_t cap) : q(cap) {}
};

#define SCHEDULER_TEMPLATE \
  template <typename TensorType, typename OptimizerType>
#define SCHEDULER_TEMPLATE_ARGS TensorType, OptimizerType
/**
 * @brief Scheduler class for Tucker decomposition.
 *
 * This class is responsible for scheduling the tensor data and optimizer tasks
 * across multiple GPU devices during Tucker decomposition. It partitions the
 * tensor data and assigns tasks to different GPUs based on the chosen
 * partitioning strategy.
 *
 * @tparam TensorType The type of tensor being decomposed.
 * @tparam OptimizerType The type of optimizer used for partitioning and task
 * scheduling.
 *
 * @details The Scheduler class manages the distribution of computation tasks
 * across GPUs, either by partitioning based on non-zero elements or by tensor
 * dimensions. It ensures that each GPU is effectively utilized for the
 * decomposition process.
 *
 * @version 1.0.0
 * @date 2023-08-10
 *
 */
SCHEDULER_TEMPLATE
class Scheduler {
  using tensor_t = TensorType;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;
  using optimizer_t = OptimizerType;

 public:
  /**
   * @brief Default constructor for the Scheduler.
   */
  explicit Scheduler(int new_gpu_count)
      : task_count(0), gpu_count(new_gpu_count) {}
  Scheduler() : task_count(0), gpu_count(0) {}
  /**
   * @brief Destructor for the Scheduler.
   */
  ~Scheduler() {}

  /**
   * @brief Schedules the tensor data and optimizer tasks across GPUs.
   *
   * This function partitions the tensor data based on the optimizer's strategy
   * and schedules the tasks across the available GPUs.
   *
   * @param tensor The tensor to be decomposed.
   * @param optimizer The optimizer that determines the partitioning strategy.
   */
  void Schedule(tensor_t* tensor, optimizer_t* optimizer);

 private:
  /**
   * @brief Partitions the tensor based on the number of non-zero elements.
   *
   * This method divides the tensor into blocks, assigning each block to a GPU
   * based on the distribution of non-zero elements.
   *
   * @param tensor The tensor to be partitioned.
   * @param optimizer The optimizer that determines the partitioning strategy.
   */
  void _NonzeroBasedPartitioning(tensor_t* tensor, optimizer_t* optimizer);

  /**
   * @brief Partitions the tensor based on its dimensions.
   *
   * This method divides the tensor into blocks, assigning each block to a GPU
   * based on the tensor's dimensions.
   *
   * @param tensor The tensor to be partitioned.
   * @param optimizer The optimizer that determines the partitioning strategy.
   */
  void _DimensionBasedPartitioning(tensor_t* tensor, optimizer_t* optimizer);

 public:
  /**
   * @brief Represents a task assigned to a GPU.
   *
   * A task includes the block ID, the number of non-zero elements, an offset
   * indicating the start position of the data, and the stream offset for CUDA
   * streams.
   */
  struct Task {
    Task() = default;
    uint64_t block_id;   ///< ID of the tensor block.
    uint64_t nnz_count;  ///< Number of non-zero elements in the block.
    uint64_t offset;     ///< Offset for the data within the block.
    uint64_t delta_offset{
        0};  ///< Offset for the delta array (global within parent block).
    int stream_offset;  ///< Stream offset for CUDA operations.
    int child_idx{-1};  ///< Index of the child block (if applicable).

    /**
     * @brief Constructs a new task.
     *
     * @param new_block_id The ID of the tensor block.
     * @param new_nnz_count The number of non-zero elements in the block.
     * @param new_offset The data offset within the block.
     * @param new_stream_offset The stream offset for CUDA operations.
     * @param new_child_idx The index of the child block.
     * @param new_delta_offset The offset for the delta array.
     */
    Task(uint64_t new_block_id, uint64_t new_nnz_count, uint64_t new_offset,
         int new_stream_offset, int new_child_idx = -1,
         uint64_t new_delta_offset = 0)
        : block_id(new_block_id),
          nnz_count(new_nnz_count),
          offset(new_offset),
          stream_offset(new_stream_offset),
          child_idx(new_child_idx),
          delta_offset(new_delta_offset) {}

    /**
     * @brief Prints the task details to the console.
     */
    void ToString() const {
      printf(
          "[%lu]-block (child: %d) \t %lu nnzs \t %lu offset \t %lu "
          "delta_offset \t stream hint %d\n",
          block_id, child_idx, nnz_count, offset, delta_offset, stream_offset);
    }
  };

 public:
  std::vector<Task> tasks;  ///< Collection of scheduled tasks.
  uint64_t task_count;  ///< The total number of tasks (may be greater than the
                        ///< block count).
  unsigned short gpu_count;  ///< The number of GPUs available for scheduling.
  uint64_t nnz_count_per_task;  ///< The number of non-zero elements per task.
};

}  // namespace gsptucker
}  // namespace supertensor

#include "gsptucker/scheduler.tpp"

#endif /* SCHEDULER_HPP_ */