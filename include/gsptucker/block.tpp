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

// Purpose: Source file for Block class
#include <cassert>

#include "gsptucker/block.hpp"
#include "gsptucker/constants.hpp"
#include "gsptucker/helper.hpp"

namespace supertensor {
namespace gsptucker {
/**
 * @brief Default constructor
 * @details Construct a new Block object
 * @return A new Block object
 *
 */
BLOCK_TEMPLATE
Block<BLOCK_TEMPLATE_ARGS>::Block() : Block(0, NULL, 0, NULL, 0) {}

/**
 * @brief Construct a new Block object
 * @details Construct a new Block object
 * @param new_block_id Block ID
 * @param new_order Block order
 * @throws std::runtime_error if the block order is less than 1
 * @return A new Block object
 */
BLOCK_TEMPLATE
Block<BLOCK_TEMPLATE_ARGS>::Block(uint64_t new_block_id,
                                  unsigned short new_order)
    : Block(new_block_id, NULL, new_order, NULL, 0) {}

/**
 * @brief Construct a new Block object
 * @details Construct a new Block object
 * @param new_block_id Block ID
 * @param new_block_coord Block coordinate
 * @param new_order Block order
 * @param new_dims Block dimensions
 * @param new_nnz_count Number of non-zero elements
 * @throws std::runtime_error if the block order is less than 1
 * @return A new Block object
 *
 */
BLOCK_TEMPLATE
Block<BLOCK_TEMPLATE_ARGS>::Block(uint64_t new_block_id,
                                  index_t* new_block_coord,
                                  unsigned short new_order, index_t* new_dims,
                                  uint64_t new_nnz_count) {
  if (new_order < 1) {
    throw std::runtime_error(
        ERROR_LOG("[ERROR] Block order should be larger than 1."));
  }
  order = new_order;
  dims = gsptucker::allocate<index_t>(this->order);
  nnz_count = new_nnz_count;
  this->_block_id = new_block_id;
  this->_base_dims = gsptucker::allocate<index_t>(this->order);
  this->_block_coord = gsptucker::allocate<index_t>(this->order);

  for (int axis = 0; axis < order; ++axis) {
    dims[axis] = new_dims[axis];
    this->_block_coord[axis] = new_block_coord[axis];  // for setting base_dims
    this->_base_dims[axis] = dims[axis] * new_block_coord[axis];
    count_nnz[axis] = nullptr;
    where_nnz[axis] = nullptr;
  }

  this->_is_allocated = false;
}

/**
 * @brief Destructor
 * @details Destroy the Block object
 *
 */
BLOCK_TEMPLATE
Block<BLOCK_TEMPLATE_ARGS>::~Block() {
  gsptucker::deallocate<index_t>(dims);
  gsptucker::deallocate<index_t>(this->_base_dims);
  gsptucker::deallocate<index_t>(this->_block_coord);
  for (unsigned short axis = 0; axis < order; ++axis) {
    gsptucker::deallocate<index_t>(indices[axis]);
    if (count_nnz[axis] != nullptr) {
      gsptucker::deallocate<uint64_t>(count_nnz[axis]);
    }
    if (where_nnz[axis] != nullptr) {
      gsptucker::deallocate<index_t>(where_nnz[axis]);
    }
  }
  gsptucker::deallocate<value_t>(values);
  this->_is_allocated = false;
  nnz_count = 0;
}

/**
 * @brief Insert a new non-zero element
 * @details Insert a new non-zero element
 * @param pos Position to insert
 * @param new_coord New coordinate
 * @param new_value New value
 *
 */
BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::InsertNonzero(uint64_t pos, index_t* new_coord,
                                               value_t new_value) {
  assert(pos <= nnz_count);
  for (unsigned short axis = 0; axis < order; ++axis) {
    indices[axis][nnz_count - pos] = new_coord[axis] - this->_base_dims[axis];
  }
  values[nnz_count - pos] = new_value;
}

/**
 * @brief Assign indices to each mode of the block
 * @details  This function processes a block tensor and assigns indices to each
 * mode based on the number of non-zero elements in each mode. The function
 * initializes temporary arrays, counts non-zero elements, and calculates the
 * starting indices for blocks. Finally, it records where each non-zero element
 * is located within the tensor.
 */
BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::AssignIndicesToEachMode() {
  assert(indices != NULL);

  uint64_t* temp_nnz[gsptucker::constants::kMaxOrder];

  // Loop through each axis and allocate memory for temporary variables to store
  // the number of non-zero elements, as well as where they are located
  for (unsigned short axis = 0; axis < order; ++axis) {
    temp_nnz[axis] = gsptucker::allocate<uint64_t>((this->dims[axis] + 1));
    count_nnz[axis] = gsptucker::allocate<uint64_t>((this->dims[axis] + 1));
    where_nnz[axis] = gsptucker::allocate<index_t>(this->nnz_count);
  }

  // Loop through each axis and initialize temporary arrays to zero
  for (unsigned short axis = 0; axis < order; ++axis) {
    for (index_t k = 0; k < this->dims[axis]; ++k) {
      count_nnz[axis][k] = 0;
      temp_nnz[axis][k] = 0;
    }
  }

  // Loop through each axis and non-zero element,
  // and count the number of non-zero elements for each axis
  for (unsigned short axis = 0; axis < order; ++axis) {
    for (uint64_t nnz = 0; nnz < this->nnz_count; ++nnz) {
      index_t k = this->indices[axis][nnz];
      assert(k < dims[axis]);
      count_nnz[axis][k]++;
      temp_nnz[axis][k]++;
    }
  }

  index_t now = 0;
  index_t k;
  index_t j = 0;

  // Loop through each axis and calculate the starting index of each block
  for (unsigned short axis = 0; axis < order; ++axis) {
    now = 0;
    uint64_t max_count = 0;

    for (j = 0; j < dims[axis]; ++j) {
      k = count_nnz[axis][j];
      if (max_count < k) {
        max_count = k;
      }
      count_nnz[axis][j] = now;
      temp_nnz[axis][j] = now;
      now += k;
    }

    count_nnz[axis][j] = now;
    temp_nnz[axis][j] = now;
  }

  // Loop through each axis and non-zero element,
  // and store where each non-zero element is located
  for (unsigned short axis = 0; axis < order; ++axis) {
    uint64_t sum_idx = 0;
    for (uint64_t nnz = 0; nnz < nnz_count; ++nnz) {
      k = indices[axis][nnz];
      now = temp_nnz[axis][k];
      where_nnz[axis][now] = nnz;
      temp_nnz[axis][k]++;
      sum_idx += k;
    }
  }
  // Deallocates
  for (unsigned short axis = 0; axis < order; ++axis) {
    gsptucker::deallocate<uint64_t>(temp_nnz[axis]);
  }
}

/**
 * @brief Allocates memory for the tensor's indices and values
 * @details This function allocates memory necessary for storing the indices and
 * values of non-zero elements in the block tensor across all its modes. It
 * ensures that the memory is only allocated if it has not already been
 * allocated.
 */
BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::AllocateData() {
  assert(nnz_count != 0);
  assert(this->_is_allocated == false);

  // Allocate memory for indices in each axis
  for (unsigned short axis = 0; axis < order; ++axis) {
    indices[axis] = gsptucker::allocate<index_t>(nnz_count);
  }
  // Allocate memory for values
  values = gsptucker::allocate<value_t>(nnz_count);
  this->_is_allocated = true;
}

BLOCK_TEMPLATE
uint64_t Block<BLOCK_TEMPLATE_ARGS>::Serialize(char* buffer) {
  uint64_t offset = 0;

  memcpy(buffer + offset, this->values, this->nnz_count * sizeof(value_t));
  offset += this->nnz_count * sizeof(value_t);
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    memcpy(buffer + offset, this->indices[axis],
           this->nnz_count * sizeof(index_t));
    offset += this->nnz_count * sizeof(index_t);
  }
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    memcpy(buffer + offset, this->where_nnz[axis],
           this->nnz_count * sizeof(index_t));
    offset += this->nnz_count * sizeof(index_t);
  }
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    memcpy(buffer + offset, this->count_nnz[axis],
           (this->dims[axis] + 1) * sizeof(uint64_t));
    offset += (this->dims[axis] + 1) * sizeof(uint64_t);
  }

  return offset;
}

BLOCK_TEMPLATE
uint64_t Block<BLOCK_TEMPLATE_ARGS>::Deserialize(char* buffer) {
  uint64_t offset = 0;

  this->values = reinterpret_cast<value_t*>(buffer + offset);
  offset += this->nnz_count * sizeof(value_t);

  for (unsigned short axis = 0; axis < this->order; ++axis) {
    this->indices[axis] = reinterpret_cast<index_t*>(buffer + offset);
    offset += this->nnz_count * sizeof(index_t);
  }

  for (unsigned short axis = 0; axis < this->order; ++axis) {
    this->where_nnz[axis] = reinterpret_cast<index_t*>(buffer + offset);
    offset += this->nnz_count * sizeof(index_t);
  }

  for (unsigned short axis = 0; axis < this->order; ++axis) {
    this->count_nnz[axis] = reinterpret_cast<uint64_t*>(buffer + offset);
    offset += (this->dims[axis] + 1) * sizeof(uint64_t);
  }

  this->_is_allocated = true;
  this->_record_ptr = buffer;

  return offset;
}

BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::DeallocateRecord() {
  if (!this->_is_allocated) {
    return;
  }

  if (indices != NULL) {
    for (unsigned short axis = 0; axis < order; ++axis) {
      if (indices[axis] != NULL) {
        gsptucker::deallocate<index_t>(indices[axis]);
        indices[axis] = NULL;
      }
    }
  }
  if (values != NULL) {
    gsptucker::deallocate<value_t>(values);
    values = NULL;
  }

  if (count_nnz != NULL) {
    for (unsigned short axis = 0; axis < order; ++axis) {
      if (count_nnz[axis] != NULL) {
        gsptucker::deallocate<uint64_t>(count_nnz[axis]);
        count_nnz[axis] = NULL;
      }
    }
  }

  if (where_nnz != NULL) {
    for (unsigned short axis = 0; axis < order; ++axis) {
      if (where_nnz[axis] != NULL) {
        gsptucker::deallocate<index_t>(where_nnz[axis]);
        where_nnz[axis] = NULL;
      }
    }
  }
  this->_is_allocated = false;
}

BLOCK_TEMPLATE
uint64_t Block<BLOCK_TEMPLATE_ARGS>::GetRecordSize() {
  uint64_t record_size = 0;
  record_size += this->nnz_count * sizeof(value_t);
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    record_size += this->nnz_count * sizeof(index_t);
  }
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    record_size += this->nnz_count * sizeof(index_t);
  }
  for (unsigned short axis = 0; axis < this->order; ++axis) {
    record_size += (this->dims[axis] + 1) * sizeof(uint64_t);
  }

  return record_size;
}

BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::ToString() {
  printf("********** BLOCK[%lu] Information **********\n", this->_block_id);

  int axis;

  printf("Block coord: ");
  for (axis = 0; axis < order; ++axis) {
    printf("[%lu]", this->_block_coord[axis]);
  }
  printf("\n");

  printf("Block order: %d\n", order);

  printf("Block dims: ");
  for (axis = 0; axis < order; ++axis) {
    printf("%lu", dims[axis]);
    if (axis < order - 1) {
      printf(" X ");
    } else {
      printf("\n");
    }
  }

  printf("# nnzs: %lu\n", nnz_count);
  PrintLine();
}

/**
 * @brief Set the dimensions of the block
 * @details This function sets the dimensions of the block based on the input
 * dimensions.
 */
BLOCK_TEMPLATE
void Block<BLOCK_TEMPLATE_ARGS>::set_dims(index_t* new_dims) {
  for (int axis = 0; axis < order; ++axis) {
    dims[axis] = new_dims[axis];
    this->_base_dims[axis] = dims[axis] * this->_block_coord[axis];
  }
}

}  // namespace gsptucker
}  // namespace supertensor