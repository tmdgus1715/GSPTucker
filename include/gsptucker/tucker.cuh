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

#ifndef TUCKER_CUH_
#define TUCKER_CUH_

#include <omp.h>

#include "common/cuda_helper.hpp"
#include "common/memory_monitor.hpp"
#include "gsptucker/constants.hpp"
#include "gsptucker/cuda_agent.hpp"
#include "gsptucker/optimizer.hpp"
#include "gsptucker/scheduler.hpp"
#include "gsptucker/tensor_manager.hpp"
#include "gsptucker/reconstruction.cuh"
#include "gsptucker/update.cuh"
// #include "gsptucker/baseline/update_sync.cuh"
// #include "gsptucker/baseline/reconstruction_sync.cuh"

namespace supertensor {
namespace gsptucker {

template <typename TensorType, typename IOManagerType>
void TuckerDecomposition(TensorType* input_tensor, int tucker_rank,
                         int gpu_count, int host_mem_limit,
                         IOManagerType* io_manager,
                         int cuda_stream_count = 1,
                         bool use_avg_partition = false) {
  using tensor_t = TensorType;
  using index_t = typename tensor_t::index_t;
  using value_t = typename tensor_t::value_t;
  using block_t = typename tensor_t::block_t;

  using optimizer_t = Optimizer<tensor_t>;
  using scheduler_t = Scheduler<tensor_t, optimizer_t>;

  using cuda_agent_t = CudaAgent<tensor_t>;
  using host_agent_t = HostAgent;

  using io_manager_t = IOManagerType;
  using tensor_manager_t = TensorManager<tensor_t, host_agent_t, io_manager_t>;

  MYPRINT("\t... Initialize HOST Agents\n");

  size_t avail_host_mem = common::GiB(host_mem_limit);
  host_agent_t* host_agent = new host_agent_t(avail_host_mem);
  std::cout << "Available Host memory: "
            << common::HumanReadable{(uintmax_t)avail_host_mem} << std::endl;

  /* Initialize Cuda Agents */
  MYPRINT("\t... Initialize CUDA Agents\n");

  cuda_agent_t** cuda_agents = allocate<cuda_agent_t*>(gpu_count);
  for (unsigned dev_id = 0; dev_id < gpu_count; ++dev_id) {
    cuda_agents[dev_id] = new cuda_agent_t(dev_id);
    cuda_agents[dev_id]->AllocateMaximumBuffer();
  }

  size_t avail_gpu_mem = cuda_agents[0]->get_allocated_size();
  std::cout << "Available GPU memory: "
            << common::HumanReadable{(uintmax_t)avail_gpu_mem} << std::endl;

  // Find optimal partition parameters from optimizer
  optimizer_t* optimizer = new optimizer_t;
  optimizer->Initialize(gpu_count, tucker_rank, avail_gpu_mem, input_tensor, cuda_stream_count);
  optimizer->use_avg_partition = use_avg_partition;
  index_t* partition_dims = optimizer->FindPartitionParms();

  // Create tensor blocks ( = sub-tensors )
  tensor_t* tensor = new tensor_t(input_tensor);
  io_manager->template CreateTensorBlocks<optimizer_t>(&input_tensor, &tensor,
                                                       optimizer);
  
  common::MemoryMonitor mem_monitor;
  mem_monitor.Start();

  tensor->ToString();

  MYPRINT("\t... Initialize Tensor Manager\n");
  tensor_manager_t* tensor_manager =
      new tensor_manager_t(host_agent, io_manager, tensor->block_count);

  MYPRINT("\t... Initialize Scheduler\n");
  scheduler_t* scheduler = new scheduler_t(gpu_count);
  scheduler->Schedule(tensor, optimizer);

  unsigned short order = tensor->order;
  index_t* dims = tensor->dims;
  index_t* block_dims = tensor->block_dims;

  MYPRINT("... Ready to fill in the factor matrices and the core tensor\n");
  value_t** factor_matrices[gsptucker::constants::kMaxOrder];

  printf("\t... Make the factor matrices\n");
  // Allocate sub_factor matrices
  for (int axis = 0; axis < order; ++axis) {
    factor_matrices[axis] =
        host_agent->PinnedAllocate<value_t*>(partition_dims[axis]);
    index_t sub_factor_row = block_dims[axis];
    for (index_t part = 0; part < partition_dims[axis]; ++part) {
      factor_matrices[axis][part] =
          host_agent->PinnedAllocate<value_t>(sub_factor_row * tucker_rank);
    }
  }

  // Initialize sub_factor matrices
  for (int axis = 0; axis < order; ++axis) {
    printf("\t\t... Fill the factor matrix [%d]\n", axis);
    index_t sub_factor_row = block_dims[axis];
    for (index_t part = 0; part < partition_dims[axis]; ++part) {
      if (part + 1 == partition_dims[axis]) {
        sub_factor_row = dims[axis] - part * block_dims[axis];
      }
      for (index_t row = 0; row < sub_factor_row; ++row) {
        for (int col = 0; col < tucker_rank; ++col) {
          factor_matrices[axis][part][row * tucker_rank + col] =
              gsptucker::frand<double>(0, 1);
        }
      }
    }
  }

  // Core tensor
  printf("\t... Make the core tensor\n");
  tensor_t* core_tensor = new tensor_t(order);
  index_t* core_dims = host_agent->Allocate<index_t>(order);
  index_t* core_part_dims = host_agent->Allocate<index_t>(order);
  uint64_t core_nnz_count = 1;
  for (int axis = 0; axis < order; ++axis) {
    core_dims[axis] = tucker_rank;
    core_part_dims[axis] = 1;
    core_nnz_count *= tucker_rank;
  }

  core_tensor->set_dims(core_dims);
  core_tensor->set_nnz_count(core_nnz_count);
  core_tensor->MakeBlocks(1, &core_nnz_count);

  block_t* curr_block = core_tensor->blocks[0];

  // #pragma omp parallel for
  for (uint64_t i = 0; i < core_nnz_count; ++i) {
    curr_block->values[i] = gsptucker::frand<double>(0, 1);
    index_t mult = 1;
    for (short axis = order; --axis >= 0;) {
      index_t idx = 0;
      if (axis == order - 1) {
        idx = i % core_dims[axis];
      } else if (axis == 0) {
        idx = i / mult;
      } else {
        idx = (i / mult) % core_dims[axis];
      }
      curr_block->indices[axis][i] = idx;
      mult *= core_dims[axis];
    }
    assert(mult == core_nnz_count);
  }
  printf("\t... Initialize the intermediate data (delta, B and C, errorT)\n");
  const uint64_t block_count = tensor->block_count;
  const index_t max_block_dim = tensor->get_max_block_dim();
  const index_t max_partition_dim = tensor->get_max_partition_dim();
  const uint64_t max_nnz_count = tensor->get_max_nnz_count_in_block();

  using matrix_t = double;
  value_t** delta = host_agent->PinnedAllocate<value_t*>(block_count);
  matrix_t* B =
      host_agent->Allocate<matrix_t>(max_block_dim * tucker_rank * tucker_rank);
  matrix_t* C = host_agent->Allocate<matrix_t>(max_block_dim * tucker_rank);
  value_t** error_T = host_agent->PinnedAllocate<value_t*>(block_count);

  for (uint64_t block_id = 0; block_id < block_count; ++block_id) {
    block_t* curr_block = tensor->blocks[block_id];
    uint64_t total_nnz = curr_block->nnz_count;
    error_T[block_id] = host_agent->PinnedAllocate<value_t>(total_nnz);
  }
  
  tensor_manager->SetTensor(tensor);
  tensor_manager->SetDelta(delta);
  tensor_manager->SetRank(tucker_rank);

  std::cout << "\t... Used size in host memory: "
            << common::HumanReadable{(uintmax_t)host_agent->GetMemoryUsage()}
            << " / "
            << common::HumanReadable{(uintmax_t)host_agent->GetMemoryLimit()}
            << std::endl;

  for (unsigned dev_id = 0; dev_id < gpu_count; ++dev_id) {
    cuda_agents[dev_id]->set_stream_count(optimizer->cuda_stream_count);
  }

  int iter = 0;
  double p_fit = -1;
  double fit = -1;

  double avg_time = omp_get_wtime();

  while (1) {
    double itertime = omp_get_wtime(), steptime;
    steptime = itertime;
    gsptucker::UpdateFactorMatrices<tensor_t, matrix_t, value_t, cuda_agent_t,
                                    scheduler_t, tensor_manager_t>(
        tensor, core_tensor, factor_matrices, delta, B, C, tucker_rank,
        gpu_count, cuda_agents, scheduler, tensor_manager);
    printf("Factor Time : %lf\n", omp_get_wtime() - steptime);

    steptime = omp_get_wtime();
    gsptucker::Reconstruction<tensor_t, value_t, value_t, cuda_agent_t,
                              scheduler_t, tensor_manager_t>(
        tensor, core_tensor, factor_matrices, &fit, error_T, tucker_rank,
        gpu_count, cuda_agents, scheduler, tensor_manager);
    printf("Recon Time : %lf\n\n", omp_get_wtime() - steptime);
    steptime = omp_get_wtime();

    ++iter;

    std::cout << "iter " << iter << "\t Fit: " << fit << std::endl;
    printf("iter%d :      Fit : %lf\tElapsed Time : %lf\n\n", iter, fit,
           omp_get_wtime() - itertime);
    if (iter >= gsptucker::constants::kMaxIteration ||
        (p_fit != -1 && gsptucker::abs<double>(p_fit - fit) <=
                            gsptucker::constants::kLambda)) {
      break;
    }
    p_fit = fit;
  }

  size_t peak_mem = mem_monitor.Stop();
  std::cout << "Peak Memory Usage: " 
            << common::HumanReadable{(uintmax_t)peak_mem * 1024} << std::endl;

  MYPRINT("DONE\n");
}

}  // namespace gsptucker
}  // namespace supertensor

#endif /* TUCKER_CUH_ */