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

#ifndef BLOCK_HPP_
#define BLOCK_HPP_

#include <cstdint>
#include <iostream>
#include <vector>

#include "gsptucker/constants.hpp"
namespace supertensor {
namespace gsptucker {

#define BLOCK_TEMPLATE template <typename IndexType, typename ValueType>
#define BLOCK_TEMPLATE_ARGS IndexType, ValueType

BLOCK_TEMPLATE
class Block {
 public:
  using this_t =
      Block<BLOCK_TEMPLATE_ARGS>;  ///< Type alias for the Block class.
  using index_t = IndexType;       ///< Type alias for the index type.
  using value_t = ValueType;       ///< Type alias for the value type.

  Block();

  Block(uint64_t new_block_id, unsigned short new_order);

  Block(uint64_t new_block_id, index_t* new_block_coord,
        unsigned short new_order, index_t* new_dims, uint64_t new_nnz_count);

  ~Block();

  bool IsEmpty() { return this->nnz_count == 0; }

  bool IsAllocated() { return this->_is_allocated; }

  void AllocateData();

  void InsertNonzero(uint64_t pos, index_t* new_coord, value_t new_value);

  void AssignIndicesToEachMode();

  uint64_t Serialize(char* buffer);

  uint64_t Deserialize(char* buffer);

  void DeallocateRecord();

  uint64_t GetRecordSize();

  void ToString();

  index_t* get_block_coord() { return this->_block_coord; }

  index_t get_block_id() { return this->_block_id; }

  index_t* get_base_dims() { return this->_base_dims; }

  char* get_record_ptr() { return this->_record_ptr; }

  void set_nnz_count(uint64_t new_nnz_count) {
    this->nnz_count = new_nnz_count;
  }

  void set_dims(index_t* new_dims);

  void set_is_allocated(bool new_is_allocated) {
    this->_is_allocated = new_is_allocated;
  }

 public:
  unsigned short order;  ///< The order (rank) of the tensor.
  index_t* dims;         ///< Dimensions of the block.
  uint64_t nnz_count;    ///< The number of non-zero elements.

  value_t* values;  ///< Array storing the values of non-zero elements.
  index_t* indices[constants::kMaxOrder];  ///< Array of pointers to indices for
                                           ///< each mode.

  index_t* where_nnz[constants::kMaxOrder];  ///< Metadata for tracking non-zero
                                             ///< element locations.
  uint64_t* count_nnz[constants::kMaxOrder];  ///< Metadata for counting
                                              ///< non-zero elements per mode.
 private:
  index_t* _base_dims;    ///< Base dimensions of the block.
  uint64_t _block_id;     ///< Unique identifier for the block.
  index_t* _block_coord;  ///< Coordinates of the block.
  bool _is_allocated;     ///< Flag indicating whether the block's data is
  char* _record_ptr;
};  // class Block

}  // namespace gsptucker
}  // namespace supertensor

#include "gsptucker/block.tpp"
#endif