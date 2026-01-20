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

#ifndef HOST_AGENT_HPP_
#define HOST_AGENT_HPP_

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

#include "common/cuda_helper.hpp"

namespace supertensor {
namespace gsptucker {
class HostAgent {
 public:
  HostAgent() : memory_usage_(0), memory_limit_(UINT64_MAX) {}
  HostAgent(uint64_t memory_limit)
      : memory_usage_(0), memory_limit_(memory_limit) {}

  template <typename T>
  T* Allocate(size_t bytes) {
    if (memory_usage_.load() + (bytes * sizeof(T)) > memory_limit_) {
      throw std::runtime_error("Memory allocation exceeds limit.");
    }

    T* ptr = gsptucker::allocate<T>(bytes);
    memory_usage_.fetch_add(bytes * sizeof(T));
    return ptr;
  }

  template <typename T>
  void Free(T* ptr, size_t bytes) {
    if (ptr) {
      gsptucker::deallocate<T>(ptr);
      memory_usage_.fetch_sub(bytes * sizeof(T));
    }
  }

  template <typename T>
  T* PinnedAllocate(size_t bytes) {
    if (memory_usage_.load() + (bytes * sizeof(T)) > memory_limit_) {
      throw std::runtime_error("Memory allocation exceeds limit.");
    }
    T* ptr = static_cast<T*>(common::cuda::pinned_malloc(sizeof(T) * bytes));
    memory_usage_.fetch_add(bytes * sizeof(T));
    return ptr;
  }

  template <typename T>
  void PinnedFree(T* ptr, size_t bytes) {
    if (ptr) {
      common::cuda::pinned_free(ptr);
      memory_usage_.fetch_sub(bytes * sizeof(T));
    }
  }

  uint64_t GetMemoryUsage() const { return memory_usage_.load(); }

  void SetMemoryLimit(uint64_t bytes) { memory_limit_ = bytes; }

  uint64_t GetMemoryLimit() const { return memory_limit_; }

  uint64_t GetAvailableMemory() const {
    return memory_limit_ - memory_usage_.load();
  }

  bool ExceedsMemoryLimit(uint64_t bytes) const {
    return memory_usage_.load() + bytes > memory_limit_;
  }

  void Reset() { memory_usage_.store(0); }

 private:
  std::atomic<uint64_t> memory_usage_;
  uint64_t memory_limit_;
};

}  // namespace gsptucker
}  // namespace supertensor

#endif  // HOST_AGENT_HPP_
