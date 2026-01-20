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

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <chrono>

#include "gsptucker/tensor_manager.hpp"

namespace supertensor {
namespace gsptucker {

TENSOR_MANAGER_TEMPLATE
TensorManager<TENSOR_MANAGER_ARGS>::TensorManager(HostAgentType* host_agent,
                                                  IOManagerType* io_manager,
                                                  uint64_t block_count)
    : host_agent_(host_agent), io_manager_(io_manager) {
  block_status.resize(block_count);
  delta_status.resize(block_count);
  for (auto& s : block_status) {
    s.Set(StatusFlag::InMemory, false);
    s.Set(StatusFlag::InStorage, true);
  }
}

TENSOR_MANAGER_TEMPLATE
typename TensorManager<TENSOR_MANAGER_ARGS>::block_t*
TensorManager<TENSOR_MANAGER_ARGS>::GetRecord(uint64_t chunk_id) {
  if (IsBlockInMemory(chunk_id)) {
    SetBlockStatus(chunk_id, StatusFlag::Borrowed, true);
    return tensor_->blocks[chunk_id];
  }

  if (!IsBlockInStorage(chunk_id)) {
    throw std::runtime_error(ERROR_LOG("Block is not in memory or storage"));
  }

  uint64_t record_size = tensor_->blocks[chunk_id]->GetRecordSize();

  if (this->host_agent_->ExceedsMemoryLimit(record_size)) {
    DumpDeltaChunks();
  }
  if (this->host_agent_->ExceedsMemoryLimit(record_size)) {
    DumpRecords();
  }
  if (this->host_agent_->ExceedsMemoryLimit(record_size)) {
    throw std::runtime_error(
        ERROR_LOG("Not enough memory to allocate record, chunk_id: " +
                  std::to_string(chunk_id)));
  }

  char* record_buffer = this->host_agent_->template Allocate<char>(record_size);
  io_manager_->ReadRecordFromStorage(chunk_id, tensor_, record_buffer);
  SetBlockStatus(chunk_id, StatusFlag::InMemory, true);
  std::cout << "Load! Chunk " << chunk_id << "\t...memory usage: "
            << common::BiM(this->host_agent_->GetMemoryUsage()) << " / "
            << common::BiM(this->host_agent_->GetMemoryLimit()) << std::endl;

  SetBlockStatus(chunk_id, StatusFlag::Borrowed, true);
  return tensor_->blocks[chunk_id];
}

TENSOR_MANAGER_TEMPLATE
typename TensorManager<TENSOR_MANAGER_ARGS>::value_t*
TensorManager<TENSOR_MANAGER_ARGS>::GetDeltaChunk(uint64_t chunk_id) {
  uint64_t nnz_count = tensor_->blocks[chunk_id]->nnz_count;
  uint64_t delta_chunk_size = nnz_count * rank_ * sizeof(value_t);

  int retry = 0;
  const int max_retry = 5;

  while (!IsDeltaInMemory(chunk_id)) {
    if (this->host_agent_->ExceedsMemoryLimit(delta_chunk_size)) {
      DumpDeltaChunks();
    }
    if (this->host_agent_->ExceedsMemoryLimit(delta_chunk_size)) {
      DumpRecords();
    }
    if (this->host_agent_->ExceedsMemoryLimit(delta_chunk_size)) {
      if (retry < max_retry) {
        ++retry;
        std::cout << "[GetDeltaChunk] Memory allocation failed, retry " << retry
                  << "/" << max_retry << " after 1s..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        continue;
      } else {
        throw std::runtime_error(ERROR_LOG(
            "Not enough memory to allocate delta chunk, chunk_id: " +
            std::to_string(chunk_id)));
      }
    }

    delta_[chunk_id] =
        this->host_agent_->template PinnedAllocate<value_t>(nnz_count * rank_);
    SetDeltaStatus(chunk_id, StatusFlag::InMemory, true);
    std::cout << "Load! DeltaChunk " << chunk_id << "\t...memory usage: "
              << common::BiM(this->host_agent_->GetMemoryUsage()) << " / "
              << common::BiM(this->host_agent_->GetMemoryLimit()) << std::endl;
  }

  SetDeltaStatus(chunk_id, StatusFlag::Borrowed, true);
  return delta_[chunk_id];
}

TENSOR_MANAGER_TEMPLATE
void TensorManager<TENSOR_MANAGER_ARGS>::DumpRecords() {
  for (uint64_t chunk_id = 0; chunk_id < this->tensor_->block_count;
       ++chunk_id) {
    if (IsBlockInMemory(chunk_id) && !IsBlockBorrowed(chunk_id)) {
      uint64_t size = this->tensor_->blocks[chunk_id]->GetRecordSize();
      this->host_agent_->template Free<char>(
          this->tensor_->blocks[chunk_id]->get_record_ptr(), size);
      SetBlockStatus(chunk_id, StatusFlag::InMemory, false);
      std::cout << "Dump! Chunk " << chunk_id << "\t...memory usage: "
                << common::BiM(this->host_agent_->GetMemoryUsage()) << " / "
                << common::BiM(this->host_agent_->GetMemoryLimit())
                << std::endl;
    }
  }
}

TENSOR_MANAGER_TEMPLATE
void TensorManager<TENSOR_MANAGER_ARGS>::DumpDeltaChunks() {
  for (uint64_t chunk_id = 0; chunk_id < this->tensor_->block_count;
       ++chunk_id) {
    if (IsDeltaInMemory(chunk_id) && !IsDeltaBorrowed(chunk_id)) {
      uint64_t nnz_count = this->tensor_->blocks[chunk_id]->nnz_count;
      uint64_t size = nnz_count * this->rank_;
      this->host_agent_->template PinnedFree<value_t>(this->delta_[chunk_id], size);
      // this->host_agent_->template Free<value_t>(this->delta_[chunk_id], size);
      SetDeltaStatus(chunk_id, StatusFlag::InMemory, false);
      std::cout << "Dump! DeltaChunk " << chunk_id << "\t...memory usage: "
                << common::BiM(this->host_agent_->GetMemoryUsage()) << " / "
                << common::BiM(this->host_agent_->GetMemoryLimit())
                << std::endl;
    }
  }
}

TENSOR_MANAGER_TEMPLATE
size_t TensorManager<TENSOR_MANAGER_ARGS>::GetTaskQueueSize(uint64_t size) {
  size_t queue_size = this->host_agent_->GetAvailableMemory() / size;

  if (queue_size < 1) {
    throw std::runtime_error(ERROR_LOG("Not enough memory to allocate queue"));
  }

  return queue_size;
}
}  // namespace gsptucker
}  // namespace supertensor