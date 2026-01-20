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

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include <cuda_runtime_api.h>

#include "common/cuda_helper.hpp"
#include "common/human_readable.hpp"
#include "common/size.hpp"

namespace common {
namespace cuda {
/**
 * @brief Get the error string
 * @details Get the error string
 * @param err Error code
 * @return Error string
 *
 */
#define _ERROR_TO_STRING(err)                                                                                                                        \
  case err:                                                                                                                                          \
    return #err;

const char *_cuda_get_error_enum(cudaError_t err) {
  switch (err) {
    _ERROR_TO_STRING(cudaSuccess)
    _ERROR_TO_STRING(cudaErrorMissingConfiguration)
    _ERROR_TO_STRING(cudaErrorMemoryAllocation)
    _ERROR_TO_STRING(cudaErrorInitializationError)
    _ERROR_TO_STRING(cudaErrorLaunchFailure)
    _ERROR_TO_STRING(cudaErrorPriorLaunchFailure)
    _ERROR_TO_STRING(cudaErrorLaunchTimeout)
    _ERROR_TO_STRING(cudaErrorLaunchOutOfResources)
    _ERROR_TO_STRING(cudaErrorInvalidDeviceFunction)
    _ERROR_TO_STRING(cudaErrorInvalidConfiguration)
    _ERROR_TO_STRING(cudaErrorInvalidDevice)
    _ERROR_TO_STRING(cudaErrorInvalidValue)
    _ERROR_TO_STRING(cudaErrorInvalidPitchValue)
    _ERROR_TO_STRING(cudaErrorInvalidSymbol)
    _ERROR_TO_STRING(cudaErrorMapBufferObjectFailed)
    _ERROR_TO_STRING(cudaErrorUnmapBufferObjectFailed)
    _ERROR_TO_STRING(cudaErrorInvalidHostPointer)
    _ERROR_TO_STRING(cudaErrorInvalidDevicePointer)
    _ERROR_TO_STRING(cudaErrorInvalidTexture)
    _ERROR_TO_STRING(cudaErrorInvalidTextureBinding)
    _ERROR_TO_STRING(cudaErrorInvalidChannelDescriptor)
    _ERROR_TO_STRING(cudaErrorInvalidMemcpyDirection)
    _ERROR_TO_STRING(cudaErrorAddressOfConstant)
    _ERROR_TO_STRING(cudaErrorTextureFetchFailed)
    _ERROR_TO_STRING(cudaErrorTextureNotBound)
    _ERROR_TO_STRING(cudaErrorSynchronizationError)
    _ERROR_TO_STRING(cudaErrorInvalidFilterSetting)
    _ERROR_TO_STRING(cudaErrorInvalidNormSetting)
    _ERROR_TO_STRING(cudaErrorMixedDeviceExecution)
    _ERROR_TO_STRING(cudaErrorCudartUnloading)
    _ERROR_TO_STRING(cudaErrorUnknown)
    _ERROR_TO_STRING(cudaErrorNotYetImplemented)
    _ERROR_TO_STRING(cudaErrorMemoryValueTooLarge)
    _ERROR_TO_STRING(cudaErrorInvalidResourceHandle)
    _ERROR_TO_STRING(cudaErrorNotReady)
    _ERROR_TO_STRING(cudaErrorInsufficientDriver)
    _ERROR_TO_STRING(cudaErrorSetOnActiveProcess)
    _ERROR_TO_STRING(cudaErrorInvalidSurface)
    _ERROR_TO_STRING(cudaErrorNoDevice)
    _ERROR_TO_STRING(cudaErrorECCUncorrectable)
    _ERROR_TO_STRING(cudaErrorSharedObjectSymbolNotFound)
    _ERROR_TO_STRING(cudaErrorSharedObjectInitFailed)
    _ERROR_TO_STRING(cudaErrorUnsupportedLimit)
    _ERROR_TO_STRING(cudaErrorDuplicateVariableName)
    _ERROR_TO_STRING(cudaErrorDuplicateTextureName)
    _ERROR_TO_STRING(cudaErrorDuplicateSurfaceName)
    _ERROR_TO_STRING(cudaErrorDevicesUnavailable)
    _ERROR_TO_STRING(cudaErrorInvalidKernelImage)
    _ERROR_TO_STRING(cudaErrorNoKernelImageForDevice)
    _ERROR_TO_STRING(cudaErrorIncompatibleDriverContext)
    _ERROR_TO_STRING(cudaErrorPeerAccessAlreadyEnabled)
    _ERROR_TO_STRING(cudaErrorPeerAccessNotEnabled)
    _ERROR_TO_STRING(cudaErrorDeviceAlreadyInUse)
    _ERROR_TO_STRING(cudaErrorProfilerDisabled)
    _ERROR_TO_STRING(cudaErrorProfilerNotInitialized)
    _ERROR_TO_STRING(cudaErrorProfilerAlreadyStarted)
    _ERROR_TO_STRING(cudaErrorProfilerAlreadyStopped)
    _ERROR_TO_STRING(cudaErrorAssert)
    _ERROR_TO_STRING(cudaErrorTooManyPeers)
    _ERROR_TO_STRING(cudaErrorHostMemoryAlreadyRegistered)
    _ERROR_TO_STRING(cudaErrorHostMemoryNotRegistered)
    _ERROR_TO_STRING(cudaErrorStartupFailure)
    _ERROR_TO_STRING(cudaErrorApiFailureBase)
  }
  return "<unknown>";
}
/**
 * @brief Check the CUDA error
 * @details Check the CUDA error
 * @param result Error code
 * @param func Function name
 *  @param file File name
 * @param line Line number
 * @return void
 */
void _cuda_check(cudaError_t result, char *const func, const char *const file, int const line) {
  if (result) {
    std::fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), _cuda_get_error_enum(result),
                 func);
    cudaDeviceReset();
    std::exit(EXIT_FAILURE);
  }
}

/**
 * @brief Allocate memory into GPU
 * @details Allocate memory into GPU
 * @param size Size of memory
 *
 */
void *device_malloc(size_t size) {
  void *p;
  _CUDA_API_CALL(cudaMalloc(&p, size));
  return p;
}
/**
 * @brief Free memory from GPU
 * @details Free memory from GPU
 * @param p Pointer to memory
 *
 */
void device_free(void *p) noexcept {
  if (p == nullptr) {
    return;
  }

  _CUDA_API_CALL(cudaFree(p));
}
/**
 * @brief Allocate pinned memory
 * @details Allocate pinned memory
 * @param size Size of memory
 *
 */
void *pinned_malloc(size_t size) {
  void *p;
  if (size == 0) {
    return nullptr;
  }
  _CUDA_API_CALL(cudaHostAlloc(&p, size, cudaHostAllocPortable));
  return p;
}
/**
 * @brief Free pinned memory
 * @details Free pinned memory
 * @param p Pointer to memory
 *
 */
void pinned_free(void *p) noexcept {
  if (p == nullptr) {
    return;
  }
  _CUDA_API_CALL(cudaFreeHost(p));
}
/**
 * @brief Get available device memory
 * @details Get available device memory
 * @return Available device memory size
 */
size_t get_available_device_memory() {
  size_t avail, total, used;
  _CUDA_API_CALL(cudaMemGetInfo(&avail, &total));
  used = total - avail;
  std::cout << "Device memory:\n\t used " << common::HumanReadable{used} << "\n\t available " << common::HumanReadable{avail} << "\n\t total "
            << common::HumanReadable{total} << std::endl;
  return avail - common::MiB(128);
}

/**
 * @brief Memcpy from host to device
 * @details Memcpy from host to device
 * @param dst Destination
 * @param src Source
 * @param size Size of memory
 *
 */
void h2dcpy(void *dst, const void *src, size_t size) { _CUDA_API_CALL(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)); }
/**
 * @brief Memcpy from host to device asynchronously
 * @details Memcpy from host to device asynchronously
 * @param dst Destination
 * @param src Source
 * @param size Size of memory
 * @param stream CUDA stream
 *
 */
void h2dcpy_async(void *dst, const void *src, size_t size, cudaStream_t const &stream) {
  _CUDA_API_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
}
/**
 * @brief Memcpy from device to host
 * @details Memcpy from device to host
 * @param dst Destination
 * @param src Source
 * @param size Size of memory
 *
 */
void d2hcpy(void *dst, const void *src, size_t size) { _CUDA_API_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)); }
/**
 * @brief Memcpy from device to host asynchronously
 * @details Memcpy from device to host asynchronously
 * @param dst Destination
 * @param src Source
 * @param size Size of memory
 * @param stream CUDA stream
 *
 */
void d2hcpy_async(void *dst, const void *src, size_t size, cudaStream_t const &stream) {
  _CUDA_API_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
}
/**
 * @brief Memcpy to symbol from device to host
 * @details Memcpy to symbol from device to host
 * @param dst Destination
 * @param src Source
 * @param size Size of memory
 *
 */
void h2dcpy_symbol(void *symbol, const void *src, size_t size) { _CUDA_API_CALL(cudaMemcpyToSymbol(symbol, src, size, 0, cudaMemcpyHostToDevice)); }

/**
 * @brief Memset on device
 * @details Memset on device
 * @param dst Destination
 * @param value Value
 * @param size Size of memory
 *
 */
void device_memset(void *dst, int value, size_t size) { _CUDA_API_CALL(cudaMemset(dst, value, size)); }
// void device_memset_async() {
// }

/* Synchronize */
/**
 * @brief Stream synchronize
 * @details Stream synchronize
 * @param stream CUDA stream
 *
 */
void stream_synchronize(cudaStream_t const &stream) { _CUDA_API_CALL(cudaStreamSynchronize(stream)); }
/**
 * @brief Device synchronize
 * @details Device synchronize
 *
 */
void device_synchronize() noexcept { _CUDA_API_CALL(cudaDeviceSynchronize()); }

/**
 * @brief Get device count
 * @details Get device count
 * @param out Number of devices
 *
 */
void device_count(int *out) {
  _CUDA_API_CALL(cudaGetDeviceCount(out));
  if (out == 0) {
    printf("There are no available device(s) that support CUDA\n");
  } else {
    printf("Detected %d CUDA Capable device(s)\n", *out);
  }
}
/**
 * @brief Set GPU device
 * @details Set GPU device
 * @param device_id Device ID
 *
 */
void set_device(int device_id) noexcept { _CUDA_API_CALL(cudaSetDevice(device_id)); }

/**
 * @brief Destroy CUDA streams
 * @details Destroy CUDA stream
 * @param stream CUDA stream
 * @param count Number of streams
 */
void destory_streams(cudaStream_t *streams, size_t count) noexcept {
  for (size_t i = 0; i < count; ++i) {
    _CUDA_API_CALL(cudaStreamDestroy(streams[i]));
  }
}

} // namespace cuda
} // namespace common
