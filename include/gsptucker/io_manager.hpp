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

#ifndef IO_MANAGER_HPP_
#define IO_MANAGER_HPP_

#include <fstream>
#include <sstream>
#include <vector>

namespace supertensor {
namespace gsptucker {

#define IO_MANAGER_TEMPLATE template <typename TensorType>
#define IO_MANAGER_ARGS TensorType

IO_MANAGER_TEMPLATE
class IOManager {
  using tensor_t = TensorType;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;
  using block_t = typename tensor_t::block_t;

 public:
  IOManager();
  IOManager(const std::string& input_path, const std::string& output_path);
  ~IOManager();

  bool ParseFromFile(tensor_t** tensor);

  bool ParseMetadataFromFile(tensor_t** tensor);

  template <typename OptimizerType>
  void CreateTensorBlocks(tensor_t** src, tensor_t** dest,
                          OptimizerType* optimizer);

  void WriteRecordToStorage(uint64_t block_id, tensor_t* tensor);

  void ReadRecordFromStorage(uint64_t chunk_id, tensor_t* tensor,
                             char* record_buffer);

 private:
  bool _ReadData(const char* buffer, const size_t buffer_length,
                 tensor_t** tensor);
  bool _ReadMetadataStream(std::ifstream& file, tensor_t** tensor,
                           size_t file_size);

  std::string _GetSSDPath(uint64_t block_id) const;

 private:
  std::string _input_path;
  std::vector<std::string> _ssd_paths;

};  // class IOManager

}  // namespace gsptucker
}  // namespace supertensor
#include "gsptucker/io_manager.tpp"
#endif  // IO_MANAGER_HPP_