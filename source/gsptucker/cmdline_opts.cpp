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

#include "gsptucker/cmdline_opts.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

#include "gsptucker/helper.hpp"
namespace supertensor {
namespace gsptucker {

CommandLineOptions::CommandLineOptions()
    : _input_path(""),
      _output_path(""),
      _order(3),
      _rank(10),
      _gpu_count(1),
      _host_memory_limit(48),
      _cuda_stream_count(1),
      _avg_partition(false) {
  Initialize();
}

CommandLineOptions::~CommandLineOptions() {}

/*
 * Initialize the command line options
 * @return void
 */
void CommandLineOptions::Initialize() {
  po::options_description options("Program Options");

  options.add_options()("help,h", "Display help menu.")(
      "input,i", po::value<std::string>(&this->_input_path),
      "Input tensor path")("output,O",
                           po::value<std::string>(&this->_output_path),
                           "Output path (SSD path)")(
      "order,o", po::value<int>(&this->_order), "Order")(
      "rank,r", po::value<int>(&this->_rank)->default_value(10), "Rank")(
      "gpus,g", po::value<int>(&this->_gpu_count)->default_value(1),
      "The number of GPUs")(
      "host_memory_limit,H",
      po::value<int>(&this->_host_memory_limit)->default_value(16),
      "Host memory limit in GB")(
      "cuda_stream_count,c", po::value<int>(&this->_cuda_stream_count)->default_value(1),
      "The number of CUDA streams")(
      "avg_partition,a", po::bool_switch(&this->_avg_partition),
      "Enable average-based partitioning");

  this->_options.add(options);
}

/*
 * Parse the command line options
 * @param argc - The number of arguments
 * @param argv - The arguments
 * @return ReturnStatus - The return status
 */
CommandLineOptions::ReturnStatus CommandLineOptions::Parse(int argc,
                                                           char* argv[]) {
  ReturnStatus ret = OPTS_SUCCESS;

  po::variables_map var_map;

  try {
    // Parse the command line options
    po::store(po::parse_command_line(argc, argv, this->_options), var_map);
    po::notify(var_map);

    // Help option
    if (var_map.count("help")) {
      std::cout << this->_options << std::endl;
      return OPTS_HELP;
    }

    // Enforce an input file argument every time
    if (!(0 < var_map.count("input"))) {
      std::cout << CYN "[ERROR] Input file must be specified!!!" RESET
                << std::endl;
      std::cout << this->_options << std::endl;
      return OPTS_FAILURE;
    } else {
      // Strip whitespaces from front/back of filename string
      boost::algorithm::trim(this->_input_path);
      ret = ValidateFile() ? OPTS_SUCCESS : OPTS_FAILURE;
    }

    // Enforce an order argument every time
    if (!(0 < var_map.count("order"))) {
      std::cout << CYN "[ERROR] Tensor order must be specified!!!" << std::endl;
      std::cout << this->_options << std::endl;
      return OPTS_FAILURE;
    }

    // We can check if a rank is defaulted
    if (!var_map["rank"].defaulted()) {
      std::cout << "[WARNING] Default value for User-Value overwritten to "
                << this->_rank << std::endl;
    }

    // We can check if the number of GPUs is defaulted
    if (!var_map["gpus"].defaulted()) {
      std::cout << "[WARNING] Default value for GPU count overwritten to "
                << this->_gpu_count << std::endl;
    }

    if (!var_map["host_memory_limit"].defaulted()) {
      std::cout
          << "[WARNING] Default value for Host Memory Limit overwritten to "
          << this->_host_memory_limit << std::endl;
    }

  } catch (std::exception& e) {
    std::cout << "[ERROR] Parsing error : " << e.what() << std::endl;
    ret = OPTS_FAILURE;
  } catch (...) {
    std::cout << "[ERROR] Parsing error (unknown type)." << std::endl;
    ret = OPTS_FAILURE;
  }

  return ret;
}

/*
 * Validate the input file
 * @return bool - True if the file is valid, false otherwise
 */
bool CommandLineOptions::ValidateFile() {
  if (!boost::filesystem::is_regular_file(this->_input_path)) {
    std::cout << CYN "[ERROR] Input file provided does not exist ["
              << this->_input_path << "]" << std::endl;
    return false;
  }
  return true;
}

}  // namespace gsptucker
}  // namespace supertensor