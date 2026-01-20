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

#ifndef HELPER_HPP_
#define HELPER_HPP_

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits> // std::remove_pointer

namespace supertensor {
namespace gsptucker {

// for Colorful print on terminal
#define RED "\x1B[31m" // red
#define GRN "\x1B[32m" // green
#define YEL "\x1B[33m" // yellow
#define BLU "\x1B[34m" // blue
#define MAG "\x1B[35m" // magenta
#define CYN "\x1B[36m" // cyan
#define WHT "\x1B[37m" // white
#define RESET "\x1B[0m"

#define MYDEBUG(Fmt, ...)                                                                                                                            \
  { printf(BLU "\t[%s] " GRN Fmt RESET, __FUNCTION__, ##__VA_ARGS__); }
#define MYDEBUG_1(Fmt, ...)                                                                                                                          \
  { printf(GRN Fmt RESET, ##__VA_ARGS__); }
#define MYPRINT(Fmt, ...)                                                                                                                            \
  { printf(CYN Fmt RESET, ##__VA_ARGS__); }

inline void PrintLine() { std::cout << "-----------------------------" << std::endl; }

/**
 * @brief Generates an error log message with file, function, and line details.
 *
 * This function creates a formatted error message that includes the file name,
 * function name, and line number where the error occurred.
 *
 * @param msg The error message.
 * @param file The file name where the error occurred.
 * @param function The function name where the error occurred.
 * @param line The line number where the error occurred.
 * @return A formatted error message string.
 */
inline std::string make_error_log(std::string msg, char const *file, char const *function, std::size_t line) {
  return std::string{"\n\n" RED} + file + "(" + std::to_string(line) + "): [" + function + "] \n\t" + msg + "\n\n" RESET;
}

#define ERROR_LOG(...) make_error_log(__VA_ARGS__, __FILE__, __func__, __LINE__)

/**
 * @brief Type alias to remove pointer from a type.
 *
 * This macro generates a type alias for a type without any pointer qualification.
 *
 * @param Type The type to remove the pointer from.
 */
#define GTUCKER_REMOVE_POINTER_TYPE_ALIAS(Type) typename std::remove_pointer<Type>::type

/**
 * @brief Allocates memory for a given type.
 *
 * Allocates memory for an array of elements of type `T`.
 *
 * @tparam T The data type of the elements.
 * @param num The number of elements to allocate.
 * @return A pointer to the allocated memory.
 * @throws std::runtime_error if memory allocation fails.
 */
template <typename T> T *allocate(size_t num) {
  T *ptr = static_cast<T *>(malloc(sizeof(T) * num));
  if (ptr == NULL) {
    throw std::runtime_error(std::string("Memory Allocation ERROR \n\t [ptr == NULL]"));
  }
  return ptr;
}

/**
 * @brief Deallocates memory.
 *
 * Frees the memory allocated for the given pointer.
 *
 * @tparam T The data type of the pointer.
 * @param ptr The pointer to the memory to deallocate.
 */
template <typename T> void deallocate(T *ptr) { free(ptr); }

/**
 * @brief Generates a random value within a specified range.
 *
 * Generates a random value between `x` and `y` of type `T`.
 *
 * @tparam T The data type of the random value.
 * @param x The lower bound of the range.
 * @param y The upper bound of the range.
 * @return A random value between `x` and `y`.
 */
template <typename T> T frand(T x, T y) { return ((y - x) * (static_cast<T>(rand()) / RAND_MAX)) + x; } // return the random value in (x, y) interval

/**
 * @brief Calculates the absolute value of a number.
 *
 * Computes the absolute value of a given number `x`.
 *
 * @tparam T The data type of the number.
 * @param x The number whose absolute value is to be calculated.
 * @return The absolute value of `x`.
 */
template <typename T> T abs(T x) { return x > 0 ? x : -x; }


} // namespace gsptucker
} // namespace supertensor
#endif // HELPER_HPP_