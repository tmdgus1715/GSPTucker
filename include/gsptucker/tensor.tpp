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

#include "gsptucker/tensor.hpp"
namespace supertensor {
namespace gsptucker {

TENSOR_TEMPLATE
Tensor<TENSOR_TEMPLATE_ARGS>::Tensor(unsigned short new_order)
    : order(new_order) {

  if (order < 1) {
    throw std::runtime_error(ERROR_LOG("[ERROR] Tensor order should be larger than 1."));
  }

  dims = gsptucker::allocate<index_t>(order);
  nnz_count = 0;
  norm = 0;

  partition_dims = gsptucker::allocate<index_t>(this->order);
  block_dims = gsptucker::allocate<index_t>(this->order);
  blocks = NULL;
  block_count = 1;

  this->_max_block_dim = 0;
  this->_max_nnz_count_in_block = 0;
  this->_empty_block_count = 0;
  this->_max_partition_dim = 1;

  for (unsigned short axis = 0; axis < this->order; ++axis) {
    dims[axis] = std::numeric_limits<index_t>::min();
    block_dims[axis] = std::numeric_limits<index_t>::min();
    partition_dims[axis] = 1; // Default partition dimension is 1
  }
}

TENSOR_TEMPLATE
Tensor<TENSOR_TEMPLATE_ARGS>::Tensor(this_t* other) : Tensor(other->order){
  set_dims(other->dims);
  nnz_count = other->nnz_count;
}

TENSOR_TEMPLATE
Tensor<TENSOR_TEMPLATE_ARGS>::Tensor() : Tensor(0) {}

TENSOR_TEMPLATE
Tensor<TENSOR_TEMPLATE_ARGS>::~Tensor() {
  gsptucker::deallocate<index_t>(dims);
  gsptucker::deallocate<index_t>(partition_dims);
  gsptucker::deallocate<index_t>(block_dims);
  if (blocks != NULL) {
    for (uint64_t block_id = 0; block_id < block_count; ++block_id) {
      delete blocks[block_id];
    }
    gsptucker::deallocate<block_t *>(blocks);
  }
}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::MakeBlocks(uint64_t new_block_count, 
                                              uint64_t *histogram) {
  assert(block_count == new_block_count);
  blocks = gsptucker::allocate<block_t *>(block_count);
  index_t *block_coord = gsptucker::allocate<index_t>(order);
  this->_empty_block_count = 0;
  this->_max_nnz_count_in_block = 0;

  for (uint64_t block_id = 0; block_id < block_count; ++block_id) {
    this->_BlockIDtoBlockCoord(block_id, block_coord);
    blocks[block_id] = new block_t(block_id, 
                                    block_coord, 
                                    order, 
                                    block_dims, 
                                    histogram[block_id]);
    blocks[block_id]->AllocateData();
    if (histogram[block_id] == 0) {
      this->_empty_block_count++;
    }
    this->_max_nnz_count_in_block = std::max<uint64_t>(histogram[block_id], 
                                                      this->_max_nnz_count_in_block);
  }
}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::InsertData(uint64_t block_id,
                                              index_t *new_indices[],
                                              value_t *new_values) {
  assert(block_id < block_count);
  // if(!blocks[block_id]->IsAllocated()) {
    for (unsigned short axis = 0; axis < order; ++axis) {
      blocks[block_id]->indices[axis] = (index_t *)new_indices[axis];
    }
    blocks[block_id]->values = (value_t *)new_values;
    blocks[block_id]->set_is_allocated(true);
  // }
}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::_BlockIDtoBlockCoord(uint64_t block_id,
                                                        index_t *coord) {
  index_t mult = 1;

  for (unsigned short iter = 0; iter < order; ++iter) {
    unsigned short axis = order - iter - 1;
    index_t idx = 0;
    if (axis == order - 1) {
      idx = block_id % partition_dims[axis];
    } else if (axis == 0) {
      idx = block_id / mult;
    } else {
      idx = (block_id / mult) % partition_dims[axis];
    }
    coord[axis] = idx;
    mult *= partition_dims[axis];
  }
}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::set_dims(index_t *new_dims) {
  for (unsigned short axis = 0; axis < order; ++axis) {
    this->dims[axis] = new_dims[axis];
  }
  this->_RefreshDims();
}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::set_partition_dims(const index_t *new_partition_dims) {
  // Initialize block dimensions
  assert(dims != NULL);
  block_count = 1;
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    partition_dims[axis] = new_partition_dims[axis];
    // Update block count
    block_count *= partition_dims[axis];
  }
  this->_RefreshDims();
}

TENSOR_TEMPLATE
uint64_t Tensor<TENSOR_TEMPLATE_ARGS>::get_max_record_size() {
  uint64_t values_size, indices_size, where_nnz_size, count_nnz_size;

  values_size = sizeof(value_t) * this->_max_nnz_count_in_block;
  indices_size = sizeof(index_t) * this->order * this->_max_nnz_count_in_block;
  where_nnz_size =
      sizeof(index_t) * this->order * this->_max_nnz_count_in_block;
  count_nnz_size = 0;
  for (ushort axis = 0; axis < this->order; ++axis) {
    count_nnz_size +=
        (this->block_dims[axis] + 1) *
        sizeof(uint64_t);  // count_nnz에서 각 모드의 사이즈는 block_dim[i] + 1
  }

  return values_size + indices_size + where_nnz_size + count_nnz_size;
}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::_RefreshDims() {
  this->_max_partition_dim = 0;
  this->_max_block_dim = 0;
  for (unsigned short axis = 0; axis < order; ++axis) {
    block_dims[axis] = (dims[axis] + partition_dims[axis] - 1) / partition_dims[axis];
    
    // Check if the block dimension is larger than the tensor dimension
    index_t check_dim = (dims[axis] + block_dims[axis] - 1) / block_dims[axis];
    if (check_dim != partition_dims[axis]) {
      throw std::runtime_error(ERROR_LOG("[ERROR] Block dimension is larger than the tensor dimension."));
    }
    this->_max_partition_dim = std::max<index_t>(this->_max_partition_dim, partition_dims[axis]);
    this->_max_block_dim = std::max<index_t>(this->_max_block_dim, block_dims[axis]);
  }
}

TENSOR_TEMPLATE
void Tensor<TENSOR_TEMPLATE_ARGS>::ToString() {
  for (unsigned short axis = 0; axis < order; ++axis) {
    printf("Max. dim[%d] = %lu\n", axis, dims[axis]);
  }
  printf("# nnzs: %lu\n", nnz_count);
  PrintLine();
  for (unsigned short axis = 0; axis < order; ++axis) {
    printf("Partition dim[%d] = %lu\n", axis, partition_dims[axis]);
  }
  printf("# blocks: %lu\n", block_count);
  PrintLine();
  for (unsigned short axis = 0; axis < order; ++axis) {
    printf("Block dim[%d] = %lu\n", axis, block_dims[axis]);
  }
  printf("# empty blocks: %lu / %lu\n", this->_empty_block_count, block_count);
}

}  // namespace gsptucker
}  // namespace supertensor