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

#include "gsptucker/constants.hpp"
#include "gsptucker/helper.hpp"
#include "gsptucker/scheduler.hpp"
#include "gsptucker/tensor.hpp"
// #include 'gsptucker/optimizer.hpp'

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

namespace supertensor {
namespace gsptucker {
/**
 * @brief Schedule the tensor data and optimizer
 * @details Schedule the tensor data and optimizer for each GPU device
 * @param tensor Tensor data
 * @param optimizer Optimizer
 *
 */
SCHEDULER_TEMPLATE
void Scheduler<SCHEDULER_TEMPLATE_ARGS>::Schedule(tensor_t* tensor,
                                                  optimizer_t* optimizer) {
  // if (optimizer->partition_type ==
  //     gsptucker::enums::PartitionTypes::kNonzeroPartition) {
  //   this->_NonzeroBasedPartitioning(tensor, optimizer);
  // } else if (optimizer->partition_type ==
  //            gsptucker::enums::PartitionTypes::kDimensionPartition) {
  this->_DimensionBasedPartitioning(tensor, optimizer);
  // } else {
  // printf("Invalid partition type\n");
  // exit(1);
  // }
}

/**
 * @brief Nonzero-based partitioning
 * @details Nonzero-based partitioning
 * @param tensor Tensor data
 * @param optimizer Optimizer
 *
 */
SCHEDULER_TEMPLATE
void Scheduler<SCHEDULER_TEMPLATE_ARGS>::_NonzeroBasedPartitioning(
    tensor_t* tensor, optimizer_t* optimizer) {
  this->tasks.clear();
  this->task_count = 0;

  uint64_t avail_nnz_count_per_task = optimizer->avail_nnz_count_per_task;
  this->nnz_count_per_task = std::min<uint64_t>(avail_nnz_count_per_task == 0
                                                    ? tensor->nnz_count
                                                    : avail_nnz_count_per_task,
                                                tensor->nnz_count);

  printf("[Scheduler] Nonzero-based partitioning (chunk size: %lu)\n",
         this->nnz_count_per_task);

  uint64_t offset = 0;
  const uint64_t block_id = 0;  // single logical block in COO ordering
  while (offset < tensor->nnz_count) {
    uint64_t remaining = tensor->nnz_count - offset;
    uint64_t chunk = std::min<uint64_t>(remaining, this->nnz_count_per_task);
    this->tasks.emplace_back(block_id, chunk, offset, -1);
    ++this->task_count;
    offset += chunk;
  }

  printf("[Scheduler] Generated %lu tasks (nonzero-based).\n",
         this->task_count);
}
/**
 * @brief Dimension-based partitioning
 * @details Dimension-based partitioning
 * @param tensor Tensor data
 * @param optimizer Optimizer
 *
 */
SCHEDULER_TEMPLATE
void Scheduler<SCHEDULER_TEMPLATE_ARGS>::_DimensionBasedPartitioning(
    tensor_t* tensor, optimizer_t* optimizer) {
  this->tasks.clear();
  this->task_count = 0;

  printf("=== Dimension-Based Partitioning ===\n");
  printf("Total blocks: %lu, GPU count: %d\n", tensor->block_count,
         this->gpu_count);

  std::vector<uint64_t> sort_nnz_count, sort_block_id;
  sort_nnz_count.resize(tensor->block_count);
  sort_block_id.resize(tensor->block_count);

  for (uint64_t block_id = 0; block_id < tensor->block_count; ++block_id) {
    sort_block_id[block_id] = block_id;
    sort_nnz_count[block_id] = tensor->blocks[block_id]->nnz_count;
  }
  std::sort(sort_block_id.begin(), sort_block_id.end(),
            [&](const uint64_t a, const uint64_t& b) {
              return (sort_nnz_count[a] > sort_nnz_count[b]);
            });
  for (uint64_t block_id = 0; block_id < tensor->block_count; ++block_id) {
    printf("[%lu] block has %lu nnzs.\n", sort_block_id[block_id],
           sort_nnz_count[sort_block_id[block_id]]);
  }

  // Dimension partitioning
  // this->nnz_count_per_task = optimizer->avail_nnz_count_per_task;
  // printf("Available nnz per task: %lu\n", this->nnz_count_per_task);

  uint64_t limit_nnz = optimizer->avail_nnz_count_per_task;

  if (optimizer->use_avg_partition) {
    uint64_t avg_nnz = (tensor->block_count > 0)
                           ? (tensor->nnz_count / tensor->block_count)
                           : 0;
    double tau = 1.0;
    uint64_t skew_nnz = avg_nnz * tau;

    this->nnz_count_per_task = std::min(skew_nnz, limit_nnz);

    printf("=== Partitioning Strategy: Average-Based ===\n");
    printf("  - Average NNZ per block: %lu\n", avg_nnz);
    printf("  - Hardware Limit (Avail NNZ): %lu\n", limit_nnz);
    printf("  - Final Task Chunk Size: %lu\n", this->nnz_count_per_task);

    if (avg_nnz > limit_nnz) {
      printf(
          "  [Info] Average NNZ exceeds hardware limit. Clamped to limit.\n");
    } else {
      printf("  [Info] Using Average NNZ for uniformity.\n");
    }
  } else {
    this->nnz_count_per_task = limit_nnz;
    printf("=== Partitioning Strategy: Hardware Limit Based ===\n");
    printf("  - Hardware Limit (Avail NNZ): %lu\n", limit_nnz);
    printf("  - Final Task Chunk Size: %lu\n", this->nnz_count_per_task);
  }

  printf("CUDA stream count per GPU: %d\n", optimizer->cuda_stream_count);

  for (uint64_t i = 0; i < tensor->block_count; ++i) {
    uint64_t block_id = sort_block_id[i];
    uint64_t remaining = tensor->blocks[block_id]->nnz_count;
    uint64_t offset = 0;

    while (remaining > 0) {
      uint64_t chunk = std::min<uint64_t>(remaining, this->nnz_count_per_task);
      this->tasks.emplace_back(block_id, chunk, offset, -1, -1, offset);
      ++this->task_count;

      index_t* block_coord = tensor->blocks[block_id]->get_block_coord();
      printf("Task: Block [%lu] (coord: ", block_id);
      for (int axis = 0; axis < tensor->order; ++axis) {
        printf("%d", block_coord[axis]);
        if (axis < tensor->order - 1) printf(",");
      }
      printf(") offset %lu nnz %lu\n", offset, chunk);

      offset += chunk;
      remaining -= chunk;
    }
  }

  printf("=== Scheduling Complete ===\n");
  printf("Total tasks created: %lu\n", this->task_count);
}

}  // namespace gsptucker
}  // namespace supertensor