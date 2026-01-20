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

#ifndef _MEMORY_REGION_H__
#define _MEMORY_REGION_H__

#include <cassert>
#include <cstdint>
#include <cstdlib>

#include "common/size.hpp"

namespace common {

#define MEMORY_REGION_TEMPLATE template <typename T>

#define MEMORY_REGION_TEMPLATE_ARGS T
/**
 * @brief MemoryRegion class for managing a contiguous block of memory.
 *
 * This class provides an abstraction for managing a region of memory, particularly useful
 * in CUDA programming where memory alignment and managing multiple streams are important.
 * It supports operations like shifting pointers, setting memory regions, and retrieving
 * memory pointers with specific offsets.
 *
 * @tparam T The data type of the elements stored in the memory region.
 *
 * @version 1.0.0
 * @date 2023-08-10
 */
MEMORY_REGION_TEMPLATE
class MemoryRegion {

public:
  using this_t = common::MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>; ///< Type alias for this class.

  /**
   * @brief Constructs a MemoryRegion with a specified pointer, size, and count.
   *
   * @param new_ptr Pointer to the base memory.
   * @param new_size Size of the memory region.
   * @param new_count Number of regions (for multi-stream CUDA applications).
   */
  MemoryRegion(void *new_ptr, uint64_t new_size, unsigned new_count);

  /**
   * @brief Constructs a MemoryRegion with a specified size and count.
   *
   * Initializes the memory region with a `nullptr` pointer.
   *
   * @param new_size Size of the memory region.
   * @param new_count Number of regions.
   */
  MemoryRegion(uint64_t new_size, unsigned new_count);

  /**
   * @brief Default constructor for MemoryRegion.
   *
   * Initializes the memory region with a `nullptr` pointer, zero size, and a count of one.
   */
  MemoryRegion();

  /**
   * @brief Default destructor for MemoryRegion.
   */
  ~MemoryRegion() = default;

  /**
   * @brief Initializes the memory region with a specified pointer, size, and count.
   *
   * @param new_ptr Pointer to the base memory.
   * @param new_size Size of the memory region.
   * @param new_count Number of regions.
   */
  void Initialize(void *new_ptr, uint64_t new_size, unsigned new_count);

  /**
   * @brief Checks if the memory region is null.
   *
   * @return `true` if the memory region's base pointer is null, `false` otherwise.
   */
  bool IsNull() const;

  /**
   * @brief Sets the memory region based on another MemoryRegion object.
   *
   * Copies the base pointer, size, and count from another MemoryRegion object.
   *
   * @param other Pointer to another MemoryRegion object.
   */
  void set_memory_region(this_t *other);

  /**
   * @brief Shifts the internal pointer by a specified size.
   *
   * The shift size is aligned to a 256-byte boundary, as commonly required by CUDA.
   *
   * @param shift_size The size by which to shift the pointer.
   */
  void shift_ptr(uint64_t shift_size);

  /**
   * @brief Sets the base pointer of the memory region.
   *
   * @param new_ptr The new base pointer.
   */
  void set_ptr(void *new_ptr);

  /**
   * @brief Sets the size of the memory region.
   *
   * The size is aligned to a 256-byte boundary.
   *
   * @param new_size The new size of the memory region.
   */
  void set_size(uint64_t new_size);

  /**
   * @brief Sets the count of memory regions.
   *
   * @param new_count The new count of regions.
   */
  void set_count(unsigned new_count);

  /**
   * @brief Retrieves a pointer to the memory region with a specified offset.
   *
   * @param offset The offset index of the region.
   * @return A pointer to the memory region at the specified offset.
   */
  T *get_ptr(unsigned offset);

  /**
   * @brief Retrieves the current shifted pointer.
   *
   * @return The current shifted pointer.
   */
  T *get_shift_ptr();

  /**
   * @brief Retrieves the total size of the memory region.
   *
   * @return The total size of the memory region in bytes.
   */
  uint64_t get_total_size() const;

  /**
   * @brief Retrieves the size of a single memory region.
   *
   * @return The size of one memory region in bytes.
   */
  uint64_t get_size() const;

  /**
   * @brief Retrieves the count of memory regions.
   *
   * @return The number of memory regions.
   */
  unsigned get_count();

private:
  T *_base_ptr;           ///< Pointer to the base memory.
  T *_shift_ptr;          ///< Pointer that tracks shifts in the base pointer.
  uint64_t _size;         ///< Size of a single memory region.
  uint64_t _aligned_size; ///< Aligned size of a single memory region (256-byte alignment).
  unsigned _count;        ///< Number of memory regions (useful for multi-stream CUDA operations).
};
/**
 * @brief Constructor
 * @details Constructor
 * @param new_ptr Pointer
 * @param new_size Size
 * @param new_count Count
 *
 */
MEMORY_REGION_TEMPLATE
MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::MemoryRegion(void *new_ptr, uint64_t new_size, unsigned new_count) {
  Initialize(new_ptr, new_size, new_count);
}
/**
 * @brief Constructor
 * @details Constructor
 * @param new_size Size
 * @param new_count Count
 *
 */
MEMORY_REGION_TEMPLATE
MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::MemoryRegion(uint64_t new_size, unsigned new_count) : MemoryRegion(NULL, new_size, new_count) {}
/**
 * @brief Constructor
 * @details Constructor
 *
 */
MEMORY_REGION_TEMPLATE
MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::MemoryRegion() : MemoryRegion(NULL, 0, 1) {}
/**
 * @brief Initialize memory region
 * @details Initialize memory region
 * @param new_ptr Pointer
 * @param new_size Size
 * @param new_count Count
 *
 */
MEMORY_REGION_TEMPLATE
void MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::Initialize(void *new_ptr, uint64_t new_size, unsigned new_count) {
  this->set_ptr(new_ptr);
  this->set_size(new_size);
  this->set_count(new_count);
}
/**
 * @brief Check if the memory region is null
 * @details Check if the memory region is null
 * @return True if the memory region is null
 */
MEMORY_REGION_TEMPLATE
bool MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::IsNull() const { return this->_base_ptr == NULL; }
/**
 * @brief Set memory region
 * @details Set memory region
 * @param other Other memory region
 *
 */
MEMORY_REGION_TEMPLATE
void MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::set_memory_region(this_t *other) {
  Initialize(other->get_ptr(0), other->get_size(), other->get_count());
}
/**
 * @brief Shift pointer
 * @details Shift pointer
 * @param shift_size Shift size
 *
 */
MEMORY_REGION_TEMPLATE
void MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::shift_ptr(uint64_t shift_size) {
  assert(this->_shift_ptr != NULL);
  uint64_t aligned_shift_size = common::aligned_size(shift_size, 256);
  void *tmp_ptr = (void *)(this->_shift_ptr);
  tmp_ptr = (void *)((uintptr_t)(tmp_ptr) + aligned_shift_size);
  this->_shift_ptr = static_cast<T *>(tmp_ptr);
}
/**
 * @brief Set pointer
 * @details Set pointer
 * @param new_ptr New pointer
 *
 */
MEMORY_REGION_TEMPLATE
void MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::set_ptr(void *new_ptr) {
  this->_base_ptr = static_cast<T *>(new_ptr);
  this->_shift_ptr = this->_base_ptr;
}
/**
 * @brief Set size
 * @details Set size
 * @param new_size New size
 *
 */
MEMORY_REGION_TEMPLATE
void MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::set_size(uint64_t new_size) {
  this->_size = new_size;
  this->_aligned_size = common::aligned_size(new_size, 256); // cudaAligmentBytes: 256
}
/**
 * @brief Set count
 * @details Set count
 * @param new_count New count
 *
 */
MEMORY_REGION_TEMPLATE
void MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::set_count(unsigned new_count) { this->_count = new_count; }
/**
 * @brief Get pointer
 * @details Get pointer
 * @param offset Offset
 * @return Pointer
 */
MEMORY_REGION_TEMPLATE
T *MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::get_ptr(unsigned offset) {
  if (IsNull()) {
    return NULL;
  }

  assert(offset < this->_count);
  void *tmp_ptr = (void *)(this->_base_ptr);
  tmp_ptr = (void *)((uintptr_t)(tmp_ptr) + this->_aligned_size * offset);
  return static_cast<T *>(tmp_ptr);
}
/**
 * @brief Get shift pointer
 * @details Get shift pointer
 * @return Shift pointer
 *
 */
MEMORY_REGION_TEMPLATE
T *MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::get_shift_ptr() {
  void *tmp_ptr = (void *)(this->_shift_ptr);
  tmp_ptr = (void *)((uintptr_t)(tmp_ptr));
  return static_cast<T *>(tmp_ptr);
}
/**
 * @brief Get total size
 * @details Get total size
 * @return Total size
 *
 */
MEMORY_REGION_TEMPLATE
uint64_t MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::get_total_size() const {
  assert(this->_aligned_size > 0);
  assert(this->_count > 0);
  return this->_aligned_size * this->_count;
}
/**
 * @brief Get size
 * @details Get size
 * @return Size
 */
MEMORY_REGION_TEMPLATE
uint64_t MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::get_size() const {
  assert(this->_aligned_size > 0);
  return this->_aligned_size;
}
/**
 * @brief Get count
 * @details Get count
 * @return Count
 *
 */
MEMORY_REGION_TEMPLATE
unsigned MemoryRegion<MEMORY_REGION_TEMPLATE_ARGS>::get_count() { return this->_count; }
} // namespace common

#endif //_MEMORY_REGION_H__