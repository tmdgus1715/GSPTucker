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

#ifndef UPDATE_CUH_
#define UPDATE_CUH_

#include <cuda_runtime_api.h>
#include <omp.h>

#include <Eigen/Dense>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include "common/cuda_helper.hpp"
#include "gsptucker/constants.hpp"
#include "gsptucker/delta.cuh"
#include "gsptucker/helper.hpp"

namespace supertensor {
namespace gsptucker {

template <typename TensorType, typename MatrixType, typename DeltaType>
void ComputingSubBC(TensorType* tensor, DeltaType* delta, MatrixType* B,
                    MatrixType* C, int curr_factor_id, int rank,
                    uint64_t block_id) {
  using block_t = typename TensorType::block_t;
  using index_t = typename TensorType::index_t;
  using value_t = typename TensorType::value_t;

  index_t* block_dims = tensor->block_dims;
  const index_t row_count = block_dims[curr_factor_id];
  const index_t* part_dims = tensor->partition_dims;

  block_t* parent_block = tensor->blocks[block_id];
  index_t* curr_block_coord = parent_block->get_block_coord();
  int part_id = curr_block_coord[curr_factor_id];
  assert(part_id < part_dims[curr_factor_id]);

  auto process_block = [&](block_t* curr_block, uint64_t delta_offset) {
    uint64_t* count_nnz = curr_block->count_nnz[curr_factor_id];
    index_t* where_nnz = curr_block->where_nnz[curr_factor_id];
    value_t* values = curr_block->values;

#pragma omp parallel for schedule(dynamic)
    for (index_t row = 0; row < row_count; ++row) {
      uint64_t nnz = count_nnz[row + 1] - count_nnz[row];
      index_t where_ptr = count_nnz[row];
      for (uint64_t kk = 0; kk < nnz; ++kk) {
        index_t pos_curr_entry = where_nnz[where_ptr + kk];
        value_t curr_entry_val = values[pos_curr_entry];

        uint64_t pos_delta = (delta_offset + pos_curr_entry) * rank;
        uint64_t pos_B = row * rank * rank;
        uint64_t pos_C = row * rank;

        for (int ii = 0; ii < rank; ++ii) {
          value_t cach = delta[block_id][pos_delta + ii];
          for (int jj = 0; jj < rank; ++jj) {
            B[pos_B++] += cach * delta[block_id][pos_delta + jj];
          }
          C[pos_C++] += cach * curr_entry_val;
        }
      }
    }
  };

  process_block(parent_block, 0);
}

template <typename IndexType, typename MatrixType>
void InitializeSubBC(MatrixType* B, MatrixType* C, IndexType row_count,
                     int rank) {
#pragma omp parallel for schedule(static)
  for (IndexType row = 0; row < row_count; ++row) {
    uint64_t pos_B = static_cast<uint64_t>(row) * rank * rank;
    uint64_t pos_C = static_cast<uint64_t>(row) * rank;
    for (int k = 0; k < rank; ++k) {
      for (int l = 0; l < rank; ++l) {
        B[pos_B] = static_cast<MatrixType>(0);
        if (k == l) {
          B[pos_B] = static_cast<MatrixType>(gsptucker::constants::kLambda);
        }
        ++pos_B;
      }
      C[pos_C] = static_cast<MatrixType>(0);
      ++pos_C;
    }
  }
}

template <typename TensorType, typename MatrixType>
void UpdateSubFactorMatrix(MatrixType*** factor_matrices, MatrixType* B,
                           MatrixType* C, int curr_factor_id, int part_id,
                           typename TensorType::index_t row_count, int rank) {
  using index_t = typename TensorType::index_t;
  using value_t = typename TensorType::value_t;
  using eigen_matrix_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;

#pragma omp parallel for schedule(static)
  for (index_t row = 0; row < row_count; ++row) {
    uint64_t pos_B = static_cast<uint64_t>(row) * rank * rank;
    uint64_t pos_C = static_cast<uint64_t>(row) * rank;
    eigen_matrix_t BB(rank, rank);
    for (int k = 0; k < rank; ++k) {
      for (int l = 0; l < rank; ++l) {
        BB(k, l) = static_cast<value_t>(
            B[pos_B + static_cast<uint64_t>(k) * rank + l]);
      }
    }
    eigen_matrix_t B_inv = BB.inverse();
    index_t offset = row * rank;
    for (int k = 0; k < rank; ++k) {
      value_t res = static_cast<value_t>(0);
      for (int l = 0; l < rank; ++l) {
        res += static_cast<value_t>(C[pos_C + l]) * B_inv(l, k);
      }
      factor_matrices[curr_factor_id][part_id][offset + k] = res;
    }
  }
}

template <typename TensorType, typename MatrixType, typename DeltaType,
          typename CudaAgentType, typename SchedulerType,
          typename TensorManagerType>
void ComputingDeltaBC(TensorType* tensor, TensorType* core_tensor,
                      MatrixType*** factor_matrices, DeltaType** delta,
                      MatrixType* B, MatrixType* C, int curr_factor_id,
                      int rank, CudaAgentType** cuda_agents,
                      SchedulerType* scheduler, int device_count,
                      TensorManagerType* tensor_manager,
                      const std::vector<uint32_t>& block_task_counts) {
  using tensor_t = TensorType;
  using block_t = typename tensor_t::block_t;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;
  using task_t = typename SchedulerType::Task;
  using queue_ctx_t = QueueCtx<task_t>;

  const index_t row_count = tensor->block_dims[curr_factor_id];
  const index_t part_count = tensor->partition_dims[curr_factor_id];
  const int part_count_int = static_cast<int>(part_count);

  uint64_t update_queue_size = 10;

  std::vector<std::unique_ptr<queue_ctx_t>> update_queues;
  update_queues.reserve(device_count);
  for (int d = 0; d < device_count; ++d)
    update_queues.emplace_back(new queue_ctx_t(update_queue_size));

  const auto& tasks = scheduler->tasks;
  if (tasks.empty()) {
    return;
  }

  const uint64_t block_count = tensor->block_count;

  std::vector<std::atomic<uint32_t>> remaining_tasks(block_count);
  std::vector<std::vector<cudaEvent_t>> block_events(block_count);
  std::vector<std::atomic<uint32_t>> event_slots(block_count);

  for (uint64_t block_id = 0; block_id < block_count; ++block_id) {
    remaining_tasks[block_id].store(block_task_counts[block_id],
                                    std::memory_order_relaxed);
    block_events[block_id].resize(block_task_counts[block_id]);
    event_slots[block_id].store(0, std::memory_order_relaxed);
  }

  std::vector<uint32_t> blocks_remaining_per_part(
      static_cast<size_t>(part_count), 0);
  for (uint64_t block_id = 0; block_id < block_count; ++block_id) {
    if (block_task_counts[block_id] > 0) {
      int p = tensor->blocks[block_id]->get_block_coord()[curr_factor_id];
      ++blocks_remaining_per_part[p];
    }
  }

  std::vector<std::unique_ptr<queue_ctx_t>> bc_queues;
  bc_queues.reserve(static_cast<size_t>(part_count));
  for (size_t part_idx = 0; part_idx < static_cast<size_t>(part_count);
       ++part_idx) {
    uint32_t capacity_hint = blocks_remaining_per_part[part_idx];
    size_t queue_capacity =
        capacity_hint > 0 ? static_cast<size_t>(capacity_hint) : 1;
    bc_queues.emplace_back(new queue_ctx_t(queue_capacity));
  }

  std::thread bc_thread([&]() {
    for (int part_id = 0; part_id < part_count_int; ++part_id) {
      auto& part_queue = *bc_queues[static_cast<size_t>(part_id)];
      auto& blocks_remaining = blocks_remaining_per_part[part_id];
      if (blocks_remaining == 0) {
        part_queue.production_done.store(true, std::memory_order_release);
        part_queue.cv_has_task.notify_all();
        continue;
      }

      while (true) {
        task_t bc_task{};
        {
          std::unique_lock<std::mutex> queue_lock(part_queue.mtx);
          part_queue.cv_has_task.wait(queue_lock, [&]() {
            return !part_queue.q.isEmpty() ||
                   part_queue.production_done.load(std::memory_order_acquire);
          });

          if (part_queue.q.isEmpty()) {
            if (part_queue.production_done.load(std::memory_order_acquire)) {
              assert(blocks_remaining == 0);
              break;
            }
            continue;
          }

          part_queue.q.dequeue(bc_task);
          queue_lock.unlock();
          part_queue.cv_has_space.notify_one();
        }

        const uint64_t block_id = bc_task.block_id;
        auto& events = block_events[block_id];
        for (cudaEvent_t& evt : events) {
          if (evt != nullptr) {
            common::cuda::_CUDA_API_CALL(cudaEventSynchronize(evt));
            common::cuda::_CUDA_API_CALL(cudaEventDestroy(evt));
            evt = nullptr;
          }
        }

        ComputingSubBC(tensor, delta, B, C, curr_factor_id, rank, block_id);

        tensor_manager->SetBlockStatus(block_id, StatusFlag::Borrowed, false);
        tensor_manager->SetDeltaStatus(block_id, StatusFlag::Borrowed, false);
        printf("\t- Finished block %lu for part %d\n", block_id, part_id);

        int block_part =
            tensor->blocks[block_id]->get_block_coord()[curr_factor_id];
        assert(block_part == part_id);
        assert(blocks_remaining > 0);
        --blocks_remaining;

        if (blocks_remaining == 0) {
          UpdateSubFactorMatrix<TensorType, MatrixType>(
              factor_matrices, B, C, curr_factor_id, part_id, row_count, rank);
          InitializeSubBC<index_t, MatrixType>(B, C, row_count, rank);

          part_queue.production_done.store(true, std::memory_order_release);
          part_queue.cv_has_task.notify_all();
          break;
        }
      }
    }
  });

  std::thread producer_thread([&]() {
    for (int part_id = 0; part_id < part_count_int; ++part_id) {
      for (const auto& task : tasks) {
        if (tensor->blocks[task.block_id]->get_block_coord()[curr_factor_id] !=
            part_id) {
          continue;
        }

        bool enqueued = false;
        while (!enqueued) {
          int target_device = 0;
          uint64_t min_load = std::numeric_limits<uint64_t>::max();
          for (int d = 0; d < device_count; ++d) {
            uint64_t load =
                update_queues[d]->pending_nnz.load(std::memory_order_relaxed);
            if (load < min_load) {
              min_load = load;
              target_device = d;
            }
          }

          auto& queue_ctx = *update_queues[target_device];
          {
            std::unique_lock<std::mutex> queue_lock(queue_ctx.mtx);
            queue_ctx.cv_has_space.wait(
                queue_lock, [&queue_ctx]() { return !queue_ctx.q.isFull(); });
          }
          tensor_manager->GetRecord(task.block_id);
          tensor_manager->GetDeltaChunk(task.block_id);
          {
            std::unique_lock<std::mutex> queue_lock(queue_ctx.mtx);
            queue_ctx.q.enqueue(task);
            queue_ctx.pending_nnz.fetch_add(task.nnz_count,
                                            std::memory_order_relaxed);
            queue_lock.unlock();
            queue_ctx.cv_has_task.notify_one();
            enqueued = true;
          }
        }
      }
    }
    for (auto& queue_ctx_ptr : update_queues) {
      queue_ctx_ptr->production_done.store(true);
      queue_ctx_ptr->cv_has_task.notify_all();
    }
  });

#pragma omp parallel num_threads(device_count)
  {
    int device_id = omp_get_thread_num();
    auto cuda_agent = cuda_agents[device_id];
    auto dev_bufs = cuda_agent->dev_buf;

    const int order = tensor->order;
    index_t* block_dims = tensor->block_dims;

    common::cuda::_CUDA_API_CALL(cudaSetDevice(device_id));
    cudaStream_t* streams =
        static_cast<cudaStream_t*>(cuda_agent->get_cuda_streams());
    int stream_count = cuda_agent->get_stream_count();

    auto dev_prof = cuda_agent->get_device_properties();
    const int max_grid_size = dev_prof->maxGridSize[0] / stream_count;

    // Set GPU device memory address
    std::uintptr_t*** h_X_idx_addr = static_cast<std::uintptr_t***>(
        common::cuda::pinned_malloc(sizeof(std::uintptr_t**) * stream_count));
    std::uintptr_t*** h_fact_addr = static_cast<std::uintptr_t***>(
        common::cuda::pinned_malloc(sizeof(std::uintptr_t**) * stream_count));
    std::uintptr_t** h_core_idx_addr =
        static_cast<std::uintptr_t**>(common::cuda::pinned_malloc(
            sizeof(std::uintptr_t*) * gsptucker::constants::kMaxOrder));

    for (int i = 0; i < stream_count; ++i) {
      h_X_idx_addr[i] =
          static_cast<std::uintptr_t**>(common::cuda::pinned_malloc(
              sizeof(std::uintptr_t*) * gsptucker::constants::kMaxOrder));
      h_fact_addr[i] =
          static_cast<std::uintptr_t**>(common::cuda::pinned_malloc(
              sizeof(std::uintptr_t*) * gsptucker::constants::kMaxOrder));
    }

    // Pre-transfer for core tensor
    for (int axis = 0; axis < order; ++axis) {
      h_core_idx_addr[axis] = reinterpret_cast<std::uintptr_t*>(
          dev_bufs.core_indices[axis].get_ptr(0));
    }
    common::cuda::h2dcpy(
        dev_bufs.core_idx_addr.get_ptr(0), h_core_idx_addr,
        sizeof(std::uintptr_t*) * gsptucker::constants::kMaxOrder);

    for (int axis = 0; axis < order; ++axis) {
      common::cuda::h2dcpy(dev_bufs.core_indices[axis].get_ptr(0),
                           core_tensor->blocks[0]->indices[axis],
                           sizeof(index_t) * core_tensor->nnz_count);
    }
    common::cuda::h2dcpy(dev_bufs.core_values.get_ptr(0),
                         core_tensor->blocks[0]->values,
                         sizeof(value_t) * core_tensor->nnz_count);

    // Computing Blocks
    int next_stream = 0;
    std::vector<uint64_t> last_block_id(stream_count, std::numeric_limits<uint64_t>::max());

    task_t queue_task{};
    auto& queue_ctx = *update_queues[device_id];

    while (true) {
      {
        std::unique_lock<std::mutex> queue_lock(queue_ctx.mtx);
        queue_ctx.cv_has_task.wait(queue_lock, [&queue_ctx]() {
          return !queue_ctx.q.isEmpty() || queue_ctx.production_done.load();
        });

        if (queue_ctx.q.isEmpty()) {
          if (queue_ctx.production_done.load()) {
            queue_lock.unlock();
            queue_ctx.cv_has_space.notify_all();
            break;
          }
          continue;
        }

        queue_ctx.q.dequeue(queue_task);
        queue_ctx.pending_nnz.fetch_sub(queue_task.nnz_count,
                                        std::memory_order_relaxed);
        queue_lock.unlock();
        queue_ctx.cv_has_space.notify_one();
      }

      uint64_t block_id = queue_task.block_id;
      uint64_t avail_nnz_count = queue_task.nnz_count;
      uint64_t nnz_offset = queue_task.offset;
      int stream_offset = next_stream;
      next_stream = (next_stream + 1) % stream_count;

      block_t* curr_block = tensor->blocks[block_id];
      index_t* curr_block_coord = curr_block->get_block_coord();

      for (int axis = 0; axis < order; ++axis) {
        h_X_idx_addr[stream_offset][axis] = reinterpret_cast<std::uintptr_t*>(
            dev_bufs.X_indices[axis].get_ptr(stream_offset));
        common::cuda::h2dcpy_async(
            dev_bufs.X_indices[axis].get_ptr(stream_offset),
            &curr_block->indices[axis][nnz_offset],
            sizeof(index_t) * avail_nnz_count, streams[stream_offset]);
        // For factor matrices
        if (last_block_id[stream_offset] != block_id) {
          h_fact_addr[stream_offset][axis] = reinterpret_cast<std::uintptr_t*>(
              dev_bufs.factors[axis].get_ptr(stream_offset));
          common::cuda::h2dcpy_async(
              dev_bufs.factors[axis].get_ptr(stream_offset),
              factor_matrices[axis][curr_block_coord[axis]],
              sizeof(value_t) * block_dims[axis] * rank,
              streams[stream_offset]);
        }
      }

      common::cuda::h2dcpy_async(
          dev_bufs.X_idx_addr.get_ptr(stream_offset),
          h_X_idx_addr[stream_offset],
          sizeof(std::uintptr_t*) * gsptucker::constants::kMaxOrder,
          streams[stream_offset]);
      if (last_block_id[stream_offset] != block_id) {
        common::cuda::h2dcpy_async(
            dev_bufs.factor_addr.get_ptr(stream_offset),
            h_fact_addr[stream_offset],
            sizeof(std::uintptr_t*) * gsptucker::constants::kMaxOrder,
            streams[stream_offset]);
        last_block_id[stream_offset] = block_id;
      }

      index_t block_size = 512;
      index_t grid_size =
          min(max_grid_size,
              max(1, (int)((avail_nnz_count + block_size - 1) / block_size)));

      dim3 blocks_per_grid(grid_size, 1, 1);
      dim3 threads_per_block(block_size, 1, 1);

      gsptucker::computing_delta_kernel<index_t, value_t>
          <<<blocks_per_grid, threads_per_block, 0, streams[stream_offset]>>>(
              (std::uintptr_t*)dev_bufs.X_idx_addr.get_ptr(stream_offset),
              (std::uintptr_t*)dev_bufs.core_idx_addr.get_ptr(0),
              (value_t*)dev_bufs.core_values.get_ptr(0),
              (value_t*)dev_bufs.delta.get_ptr(stream_offset),
              (std::uintptr_t*)dev_bufs.factor_addr.get_ptr(stream_offset),
              order, rank, curr_factor_id, avail_nnz_count,
              core_tensor->nnz_count);

      // For Delta
      uint64_t delta_offset = queue_task.delta_offset;
      // printf("DEBUG: d2hcpy_async delta[%lu][%lu] size %lu\n", block_id,
      // delta_offset * rank, sizeof(value_t) * avail_nnz_count * rank);
      common::cuda::d2hcpy_async(&delta[block_id][delta_offset * rank],
                                 dev_bufs.delta.get_ptr(stream_offset),
                                 sizeof(value_t) * avail_nnz_count * rank,
                                 streams[stream_offset]);

      cudaEvent_t copy_event;
      common::cuda::_CUDA_API_CALL(
          cudaEventCreateWithFlags(&copy_event, cudaEventDisableTiming));
      common::cuda::_CUDA_API_CALL(
          cudaEventRecord(copy_event, streams[stream_offset]));

      uint32_t event_slot =
          event_slots[block_id].fetch_add(1, std::memory_order_acq_rel);
      block_events[block_id][event_slot] = copy_event;

      uint32_t prev =
          remaining_tasks[block_id].fetch_sub(1, std::memory_order_acq_rel);
      if (prev == 1) {
        task_t bc_task{};
        bc_task.block_id = block_id;
        int block_part =
            tensor->blocks[block_id]->get_block_coord()[curr_factor_id];
        auto& part_queue = *bc_queues[static_cast<size_t>(block_part)];

        std::unique_lock<std::mutex> bc_lock(part_queue.mtx);
        part_queue.cv_has_space.wait(
            bc_lock, [&part_queue]() { return !part_queue.q.isFull(); });
        part_queue.q.enqueue(bc_task);
        bc_lock.unlock();
        part_queue.cv_has_task.notify_one();
      }
    }  // task loop
  }

  for (auto& queue_ctx_ptr : bc_queues) {
    queue_ctx_ptr->production_done.store(true, std::memory_order_release);
    queue_ctx_ptr->cv_has_task.notify_all();
  }
  if (bc_thread.joinable()) {
    bc_thread.join();
  }

  if (producer_thread.joinable()) {
    producer_thread.join();
  }
}

template <typename TensorType, typename MatrixType, typename ValueType,
          typename CudaAgentType, typename SchedulerType,
          typename TensorManagerType>
void UpdateFactorMatrices(TensorType* tensor, TensorType* core_tensor,
                          ValueType*** factor_matrices, ValueType** delta,
                          MatrixType* B, MatrixType* C, int rank,
                          int device_count, CudaAgentType** cuda_agents,
                          SchedulerType* scheduler,
                          TensorManagerType* tensor_manager) {
  using tensor_t = TensorType;
  using block_t = typename tensor_t::block_t;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;
  using matrix_t = Eigen::MatrixXd;

  for (unsigned dev_id = 0; dev_id < device_count; ++dev_id) {
    cuda_agents[dev_id]->SetDeviceBuffers(tensor, rank,
                                          scheduler->nnz_count_per_task);
  }

  for (int curr_factor_id = 0; curr_factor_id < tensor->order;
       ++curr_factor_id) {
    MYPRINT("[ Update factor matrix %d ]\n", curr_factor_id);
    // Pre-compute the number of tasks per block
    std::vector<uint32_t> block_task_counts(tensor->block_count, 0);
    for (const auto& task : scheduler->tasks) {
      ++block_task_counts[task.block_id];
    }

    int row_count = tensor->block_dims[curr_factor_id];
    // Initialize B and C
    InitializeSubBC<index_t, MatrixType>(B, C, row_count, rank);

    double update_time = omp_get_wtime();
    // Computing Delta, B, and C
    ComputingDeltaBC(tensor, core_tensor, factor_matrices, delta, B, C,
                     curr_factor_id, rank, cuda_agents, scheduler, device_count,
                     tensor_manager, block_task_counts);

    update_time = omp_get_wtime() - update_time;
    printf("\t- Elapsed time for Updateing Factor Matrix %d: %lf\n",
           curr_factor_id, update_time);

    // printf("\t- row-wise update TIME : %lf\n", update_time);

  }  // ! curr_factor
}

}  // namespace gsptucker
}  // namespace supertensor

#endif /* UPDATE_CUH_ */