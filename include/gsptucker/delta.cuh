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

#ifndef DELTA_CUH_
#define DELTA_CUH_

#include <cuda_runtime_api.h>

#include "common/cuda_helper.hpp"
#include "gsptucker/helper.hpp"

namespace supertensor {
namespace gsptucker {

template <typename IndexType, typename ValueType>
__global__ void computing_delta_kernel(std::uintptr_t* X_indices,
                                       std::uintptr_t* core_indices,
                                       ValueType* core_values, ValueType* delta,
                                       std::uintptr_t* factors, const int order,
                                       const int rank, int curr_factor_id,
                                       uint64_t nnz_count,
                                       uint64_t core_nnz_count) {
  using index_t = IndexType;
  using value_t = ValueType;

  uint64_t tid = blockDim.x * blockIdx.x + threadIdx.x;  // per #NNZs
  uint64_t stride = blockDim.x * gridDim.x;

  __shared__ int sh_rank;
  __shared__ std::uintptr_t* sh_X_idx_addr[gsptucker::constants::kMaxOrder];
  __shared__ std::uintptr_t* sh_core_idx_addr[gsptucker::constants::kMaxOrder];
  __shared__ std::uintptr_t* sh_factors[gsptucker::constants::kMaxOrder];

  if (threadIdx.x == 0) {
    sh_rank = rank;
    for (int axis = 0; axis < order; ++axis) {
      sh_X_idx_addr[axis] = reinterpret_cast<std::uintptr_t*>(X_indices[axis]);
      sh_core_idx_addr[axis] =
          reinterpret_cast<std::uintptr_t*>(core_indices[axis]);
      sh_factors[axis] = reinterpret_cast<std::uintptr_t*>(factors[axis]);
    }
  }
  __syncthreads();

  // tid == row for the delta
  int r, axis;
  uint64_t i;

  while (tid < nnz_count) {
    value_t tmp[50];
    index_t nnz[gsptucker::constants::kMaxOrder];
    for (r = 0; r < sh_rank; ++r) {
      tmp[r] = 0.0f;
    }

    for (axis = 0; axis < order; ++axis) {
      nnz[axis] = ((index_t*)sh_X_idx_addr[axis])[tid];
    }
    for (i = 0; i < core_nnz_count; ++i) {
      index_t delta_col = ((index_t*)core_indices[curr_factor_id])[i];
      value_t beta = core_values[i];
      for (axis = 0; axis < order; ++axis) {
        // int axis = (curr_factor_id + 1 + iter + order) % order;

        if (axis != curr_factor_id) {
          // index_t row = ((index_t *)X_indices[axis])[tid];
          // index_t col = ((index_t *)core_indices[axis])[i];
          beta *= ((value_t*)(sh_factors[axis]))
              [nnz[axis] * sh_rank + ((index_t*)sh_core_idx_addr[axis])[i]];
        }
      }
      tmp[delta_col] += beta;
    }
    for (r = 0; r < sh_rank; ++r) {
      delta[tid * sh_rank + r] = tmp[r];
    }
    tid += stride;
  }
  __syncthreads();
}
}  // namespace gsptucker
}  // namespace supertensor

#endif /* DELTA_CUH_ */