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

#ifndef CUDA_HELPER_HPP_
#define CUDA_HELPER_HPP_

#include <cstring>

#include <cuda_runtime_api.h>

namespace common {
namespace cuda {

/**
 * @brief Converts a CUDA error code to a string.
 *
 * This function converts a CUDA error code to its corresponding error string.
 *
 * @param err The CUDA error code.
 * @return The corresponding error string.
 */
const char *_cuda_get_error_enum(cudaError_t err);

/**
 * @brief Checks the result of a CUDA API call.
 *
 * This function checks the result of a CUDA API call and prints an error message
 * if the call failed. It includes the function name, file name, and line number
 * in the error message.
 *
 * @param result The result of the CUDA API call.
 * @param func The name of the function that made the CUDA API call.
 * @param file The name of the file containing the function.
 * @param line The line number where the CUDA API call was made.
 */
void _cuda_check(cudaError_t result, char *const func, const char *const file, int const line);
#define _CUDA_API_CALL(val) _cuda_check((val), #val, __FILE__, __LINE__)

/**
 * @brief Allocates memory on the GPU.
 *
 * This function allocates a specified amount of memory on the GPU.
 *
 * @param size The size of the memory to allocate in bytes.
 * @return A pointer to the allocated memory on the GPU.
 */
void *device_malloc(size_t size);

/**
 * @brief Frees memory on the GPU.
 *
 * This function frees memory that was previously allocated on the GPU.
 *
 * @param p A pointer to the memory to free.
 */
void device_free(void *p) noexcept;

/**
 * @brief Allocates pinned memory on the host.
 *
 * This function allocates a specified amount of pinned memory on the host.
 *
 * @param size The size of the memory to allocate in bytes.
 * @return A pointer to the allocated pinned memory.
 */
void *pinned_malloc(size_t size);

/**
 * @brief Frees pinned memory on the host.
 *
 * This function frees pinned memory that was previously allocated on the host.
 *
 * @param p A pointer to the pinned memory to free.
 */
void pinned_free(void *p) noexcept;

/**
 * @brief Gets the amount of available memory on the GPU.
 *
 * This function returns the amount of free memory available on the GPU.
 *
 * @return The amount of free memory in bytes.
 */
size_t get_available_device_memory();

/**
 * @brief Copies memory from host to device.
 *
 * This function copies a specified amount of memory from host to device.
 *
 * @param dst The destination pointer on the device.
 * @param src The source pointer on the host.
 * @param size The size of the memory to copy in bytes.
 */
void h2dcpy(void *dst, const void *src, size_t size);

/**
 * @brief Asynchronously copies memory from host to device.
 *
 * This function asynchronously copies a specified amount of memory from host to device.
 *
 * @param dst The destination pointer on the device.
 * @param src The source pointer on the host.
 * @param size The size of the memory to copy in bytes.
 * @param stream The CUDA stream to use for the asynchronous copy.
 */
void h2dcpy_async(void *dst, const void *src, size_t size, cudaStream_t const &stream);

/**
 * @brief Copies memory from device to host.
 *
 * This function copies a specified amount of memory from device to host.
 *
 * @param dst The destination pointer on the host.
 * @param src The source pointer on the device.
 * @param size The size of the memory to copy in bytes.
 */
void d2hcpy(void *dst, const void *src, size_t size);

/**
 * @brief Asynchronously copies memory from device to host.
 *
 * This function asynchronously copies a specified amount of memory from device to host.
 *
 * @param dst The destination pointer on the host.
 * @param src The source pointer on the device.
 * @param size The size of the memory to copy in bytes.
 * @param stream The CUDA stream to use for the asynchronous copy.
 */
void d2hcpy_async(void *dst, const void *src, size_t size, cudaStream_t const &stream);

/**
 * @brief Copies memory to a CUDA symbol on the device.
 *
 * This function copies a specified amount of memory to a CUDA symbol on the device.
 *
 * @param symbol The CUDA symbol on the device.
 * @param src The source pointer on the host.
 * @param size The size of the memory to copy in bytes.
 */
void h2dcpy_symbol(void *symbol, const void *src, size_t size);

/**
 * @brief Sets memory on the GPU to a specified value.
 *
 * This function sets a specified amount of memory on the GPU to a given value.
 *
 * @param dst The destination pointer on the device.
 * @param value The value to set the memory to.
 * @param size The size of the memory to set in bytes.
 */
void device_memset(void *dst, int value, size_t size);

/**
 * @brief Synchronizes a CUDA stream.
 *
 * This function synchronizes a specified CUDA stream, ensuring that all operations
 * in the stream are completed before proceeding.
 *
 * @param stream The CUDA stream to synchronize.
 */
void stream_synchronize(cudaStream_t const &stream);

/**
 * @brief Synchronizes the CUDA device.
 *
 * This function synchronizes the CUDA device, ensuring that all operations on the device
 * are completed before proceeding.
 */
void device_synchronize() noexcept;

/**
 * @brief Gets the number of CUDA devices.
 *
 * This function retrieves the number of CUDA-capable devices available on the system.
 *
 * @param out A pointer to an integer where the device count will be stored.
 */
void device_count(int *out);

/**
 * @brief Sets the current CUDA device.
 *
 * This function sets the CUDA device that will be used for subsequent CUDA operations.
 *
 * @param device_id The ID of the device to set as the current device.
 */
void set_device(int device_id) noexcept;

/**
 * @brief Destroys CUDA streams.
 *
 * This function destroys a specified number of CUDA streams.
 *
 * @param streams A pointer to the array of streams to destroy.
 * @param count The number of streams to destroy.
 */
void destory_streams(cudaStream_t *streams, size_t count) noexcept;

} // namespace cuda
} // namespace common

#endif /* CUDA_HELPER_HPP_ */