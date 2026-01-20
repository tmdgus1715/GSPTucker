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

#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include "gsptucker/block.hpp"

namespace supertensor {
namespace gsptucker {

#define TENSOR_TEMPLATE template <typename BlockType>
#define TENSOR_TEMPLATE_ARGS BlockType

TENSOR_TEMPLATE
class Tensor {
public:
  using this_t = Tensor<TENSOR_TEMPLATE_ARGS>; ///< Type alias for this tensor type.
  using block_t = BlockType;                   ///< Type alias for the block type.
  using index_t = typename block_t::index_t;   ///< Type alias for the index type.
  using value_t = typename block_t::value_t;   ///< Type alias for the value type.

public:
  Tensor(unsigned short new_order);

  Tensor(this_t *other);

  Tensor();

  ~Tensor();

  void MakeBlocks(uint64_t new_block_count, uint64_t *histogram);

  void InsertData(uint64_t block_id, index_t *indices[], value_t *values);

  void ToString();

  void set_dims(index_t *new_dims);

  void set_partition_dims(const index_t *new_partition_dims);

  void set_nnz_count(uint64_t new_nnz_count) { nnz_count = new_nnz_count; }

  index_t get_max_partition_dim() { return this->_max_partition_dim; }

  index_t get_max_block_dim() { return this->_max_block_dim; }

  index_t get_max_nnz_count_in_block() { return this->_max_nnz_count_in_block; }

  uint64_t get_max_record_size();

private:
  void _BlockIDtoBlockCoord(uint64_t block_id, index_t *coord);

  void _RefreshDims();

public:
  /* Tensor Description */
  unsigned short order; ///< The number of dimensions (order) of the tensor.
  index_t *dims;        ///< Array representing the dimensions of the tensor.
  uint64_t nnz_count;   ///< The number of non-zero elements in the tensor.
  value_t norm;         ///< The norm of the tensor.

  /* Block Description */
  index_t *partition_dims; ///< The dimensions used for partitioning the tensor.
  index_t *block_dims;     ///< The dimensions of each block in the tensor.
  uint64_t block_count;    ///< The total number of blocks in the tensor.
  block_t **blocks;        ///< Array of pointers to blocks representing sub-tensors.

private:
  uint64_t _max_nnz_count_in_block; ///< The maximum number of non-zero elements in any block.
  index_t _max_block_dim;           ///< The maximum dimension size among the blocks.
  uint64_t _empty_block_count;      ///< The count of empty blocks.
  index_t _max_partition_dim;       ///< The maximum partition dimension.
}; // class Tensor

} // namespace gsptucker
} // namespace supertensor
#include "gsptucker/tensor.tpp"
#endif /* TENSOR_HPP_ */