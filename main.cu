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

#include <iostream>

#include "gsptucker/cmdline_opts.hpp"
#include "gsptucker/io_manager.hpp"
#include "gsptucker/tensor.hpp"
#include "gsptucker/tucker.cuh"
// #include "gsptucker/baseline/tucker_nontiling.cuh"

int main(int argc, char* argv[]) {
  using namespace supertensor::gsptucker;
  CommandLineOptions* options = new CommandLineOptions;
  CommandLineOptions::ReturnStatus ret = options->Parse(argc, argv);

  if (CommandLineOptions::OPTS_SUCCESS == ret) {
    // Input file
    std::cout << options->get_input_path() << std::endl;

    using index_t = uint32_t;
    using value_t = double;
    using block_t = Block<index_t, value_t>;
    using tensor_t = Tensor<block_t>;
    using io_manager_t = IOManager<tensor_t>;

    bool is_double = std::is_same<value_t, double>::value;
    if (is_double) {
      printf("Values are double type.\n");
    } else {
      printf("Values are float type.\n");
    }

    // Read tensor from file
    io_manager_t* io_manager =
        new io_manager_t(options->get_input_path(), options->get_output_path());
    tensor_t* input_tensor = new tensor_t(options->get_order());
    io_manager->ParseFromFile(&input_tensor);

    // Perform Tucker decomposition
    TuckerDecomposition<tensor_t, io_manager_t>(
        input_tensor, options->get_rank(), options->get_gpu_count(),
        options->get_host_memory_limit(), io_manager,
        options->get_cuda_stream_count(),
        options->get_avg_partition());

  } else {
    std::cout << "ERROR - problem with options." << std::endl;
  }

  return 0;
}
