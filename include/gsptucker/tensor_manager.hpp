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

#ifndef TENSOR_MANAGER_HPP_
#define TENSOR_MANAGER_HPP_

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "common/circlequeue.hpp"
#include "common/human_readable.hpp"
#include "common/size.hpp"
#include "gsptucker/helper.hpp"
#include "gsptucker/host_agent.hpp"

namespace supertensor {
namespace gsptucker {

enum class StatusFlag { InMemory, InStorage, Borrowed };

class Status {
 public:
  Status() = default;
  Status(Status&& other) noexcept {
    in_memory.store(other.in_memory.load());
    in_storage.store(other.in_storage.load());
    is_borrowed.store(other.is_borrowed.load());
  }

  Status& operator=(Status&& other) noexcept {
    if (this != &other) {
      in_memory.store(other.in_memory.load());
      in_storage.store(other.in_storage.load());
      is_borrowed.store(other.is_borrowed.load());
    }
    return *this;
  }

  Status(const Status&) = delete;
  Status& operator=(const Status&) = delete;

  void Set(StatusFlag flag, bool value) { GetAtomic(flag).store(value); }

  bool Get(StatusFlag flag) const { return GetAtomic(flag).load(); }

 private:
  std::atomic<bool> in_memory{false};
  std::atomic<bool> in_storage{false};
  std::atomic<bool> is_borrowed{false};

  std::atomic<bool>& GetAtomic(StatusFlag flag) {
    switch (flag) {
      case StatusFlag::InMemory:
        return in_memory;
      case StatusFlag::InStorage:
        return in_storage;
      case StatusFlag::Borrowed:
        return is_borrowed;
    }
    throw std::invalid_argument("Unknown flag");
  }
    const std::atomic<bool>& GetAtomic(StatusFlag flag) const {
    switch (flag) {
      case StatusFlag::InMemory:
        return in_memory;
      case StatusFlag::InStorage:
        return in_storage;
      case StatusFlag::Borrowed:
        return is_borrowed;
    }
    throw std::invalid_argument("Unknown flag");
  }
};  

#define TENSOR_MANAGER_TEMPLATE \
  template <typename TensorType, typename HostAgentType, typename IOManagerType>
#define TENSOR_MANAGER_ARGS TensorType, HostAgentType, IOManagerType

TENSOR_MANAGER_TEMPLATE
class TensorManager {
  using tensor_t = TensorType;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;
  using block_t = typename tensor_t::block_t;
  using host_agent_t = HostAgentType;
  using io_manager_t = IOManagerType;

 public:
  TensorManager() = delete;
  TensorManager(HostAgentType *host_agent, IOManagerType *io_manager, uint64_t block_count);
  ~TensorManager();

  size_t GetTaskQueueSize(uint64_t size);
  
  block_t *GetRecord(uint64_t block_id);

  value_t *GetDeltaChunk(uint64_t block_id);

  void DumpRecords();

  void DumpDeltaChunks();

  void SetBlockStatus(uint64_t id, StatusFlag flag, bool v) {
    SetStatus(block_status, id, flag, v);
  }
  void SetDeltaStatus(uint64_t id, StatusFlag flag, bool v) {
    SetStatus(delta_status, id, flag, v);
  }

  bool IsBlockInMemory(uint64_t id) const {
    return GetStatus(block_status, id, StatusFlag::InMemory);
  }
  bool IsBlockInStorage(uint64_t id) const {
    return GetStatus(block_status, id, StatusFlag::InStorage);
  }
  bool IsBlockBorrowed(uint64_t id) const {
    return GetStatus(block_status, id, StatusFlag::Borrowed);
  }

  bool IsDeltaInMemory(uint64_t id) const {
    return GetStatus(delta_status, id, StatusFlag::InMemory);
  }
  bool IsDeltaInStorage(uint64_t id) const {
    return GetStatus(delta_status, id, StatusFlag::InStorage);
  }
  bool IsDeltaBorrowed(uint64_t id) const {
    return GetStatus(delta_status, id, StatusFlag::Borrowed);
  }

  void SetTensor(tensor_t *tensor) { tensor_ = tensor; }

  void SetDelta(value_t **delta) { delta_ = delta; }

  void SetRank(int rank) { rank_ = rank; }


 private:

  void SetStatus(std::vector<Status> &statuses, uint64_t id, StatusFlag flag,
                 bool value) {
    statuses[id].Set(flag, value);
  }

  bool GetStatus(const std::vector<Status> &statuses, uint64_t id,
                 StatusFlag flag) const {
    return statuses[id].Get(flag);
  }

 private:
  std::vector<Status> block_status;
  std::vector<Status> delta_status;

  tensor_t *tensor_;
  value_t **delta_;

  int rank_;

  host_agent_t *host_agent_;
  io_manager_t *io_manager_;
};

}  // namespace gsptucker
}  // namespace supertensor
#include "gsptucker/tensor_manager.tpp"
#endif  // TENSOR_READER_HPP_