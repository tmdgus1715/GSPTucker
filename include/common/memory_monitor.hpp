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

#ifndef MEMORY_MONITOR_HPP_
#define MEMORY_MONITOR_HPP_

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <sstream>

namespace common {

class MemoryMonitor {
 public:
  MemoryMonitor() : running_(false), peak_memory_kb_(0) {}

  ~MemoryMonitor() {
    Stop();
  }

  void Start() {
    if (running_) return;
    running_ = true;
    peak_memory_kb_ = GetCurrentMemoryUsageKB(); // Initial reading
    monitor_thread_ = std::thread(&MemoryMonitor::MonitorLoop, this);
  }

  size_t Stop() {
    if (!running_) return peak_memory_kb_;
    running_ = false;
    if (monitor_thread_.joinable()) {
      monitor_thread_.join();
    }
    return peak_memory_kb_;
  }

  size_t GetPeakMemoryKB() const {
    return peak_memory_kb_;
  }

 private:
  std::atomic<bool> running_;
  std::atomic<size_t> peak_memory_kb_;
  std::thread monitor_thread_;

  void MonitorLoop() {
    while (running_) {
      size_t current_mem = GetCurrentMemoryUsageKB();
      size_t old_peak = peak_memory_kb_.load();
      while (current_mem > old_peak && !peak_memory_kb_.compare_exchange_weak(old_peak, current_mem)) {
        // Retry if update failed
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  size_t GetCurrentMemoryUsageKB() {
    std::ifstream status_file("/proc/self/status");
    std::string line;
    size_t vm_rss = 0;
    while (std::getline(status_file, line)) {
      if (line.find("VmRSS:") != std::string::npos) {
        std::stringstream ss(line);
        std::string label;
        std::string unit;
        ss >> label >> vm_rss >> unit; // VmRSS: 1234 kB
        break;
      }
    }
    return vm_rss;
  }
};

}  // namespace common

#endif // MEMORY_MONITOR_HPP_
