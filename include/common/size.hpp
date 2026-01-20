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

#ifndef _SIZE_H__
#define _SIZE_H__

#include <cstddef>

namespace common {
/**
 * @brief Converts a size in KiB (kibibytes) to bytes.
 *
 * This function converts a size in kibibytes to its equivalent size in bytes.
 *
 * @tparam T The type of the return value (default is `size_t`).
 * @param i The size in kibibytes.
 * @return The size in bytes.
 *
 */
template <typename T = size_t>
constexpr T KiB(size_t const i) {
  return i << 10;
}

/**
 * @brief Converts a size in MiB (mebibytes) to bytes.
 *
 * This function converts a size in mebibytes to its equivalent size in bytes.
 *
 * @tparam T The type of the return value (default is `size_t`).
 * @param i The size in mebibytes.
 * @return The size in bytes.
 *
 */
template <typename T = size_t>
constexpr T MiB(size_t const i) {
  return i << 20;
}

/**
 * @brief Converts a size in GiB (gibibytes) to bytes.
 *
 * This function converts a size in gibibytes to its equivalent size in bytes.
 *
 * @tparam T The type of the return value (default is `size_t`).
 * @param i The size in gibibytes.
 * @return The size in bytes.
 *
 */
template <typename T = size_t>
constexpr T GiB(size_t const i) {
  return i << 30;
}

/**
 * @brief Converts a size in TiB (tebibytes) to bytes.
 *
 * This function converts a size in tebibytes to its equivalent size in bytes.
 *
 * @tparam T The type of the return value (default is `size_t`).
 * @param i The size in tebibytes.
 * @return The size in bytes.
 *
 */
template <typename T = size_t>
constexpr T TiB(size_t const i) {
  return i << 40;
}

template <typename T = size_t>
constexpr T BiM(size_t const i) {
  return i >> 20;
}

/**
 * @brief Calculates the aligned size based on the given alignment.
 *
 * This function returns the smallest size that is a multiple of `align` and is
 * greater than or equal to `size`.
 *
 * @tparam T1 The type of the size value.
 * @tparam T2 The type of the alignment value.
 * @param size The size to be aligned.
 * @param align The alignment boundary.
 * @return The aligned size.
 *
 */
template <typename T1, typename T2>
constexpr T1 aligned_size(const T1 size, const T2 align) {
  return align * ((size + align - 1) / align);
}

}  // namespace common
#endif  // _SIZE_H__