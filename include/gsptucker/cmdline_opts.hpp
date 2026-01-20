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

#ifndef CMDLINE_OPTS_HPP_
#define CMDLINE_OPTS_HPP_

#include <boost/program_options.hpp>
#include <cstdint>
#include <string>

namespace po = boost::program_options;

namespace supertensor {
namespace gsptucker {

/**
 * @brief Command line options for the Tucker decomposition program.
 *
 * This class defines and handles the command line options used by the Tucker
 * decomposition program. It parses command line arguments and stores essential
 * parameters such as the input path, tensor order, rank, and the number of GPUs
 * to be used.
 *
 * @author Jihye Lee
 * @date 2023-08-10
 * @version 1.0.0
 */
class CommandLineOptions {
 public:
  /**
   * @brief Enum to represent the status of command line option parsing.
   */
  enum ReturnStatus {
    OPTS_SUCCESS,  ///< Parsing was successful.
    OPTS_HELP,     ///< Help was requested.
    OPTS_FAILURE   ///< Parsing failed.
  };

  /**
   * @brief Constructor for CommandLineOptions.
   *
   * Initializes the command line options with default values.
   */
  CommandLineOptions();

  /**
   * @brief Destructor for CommandLineOptions.
   */
  ~CommandLineOptions();

  /**
   * @brief Parses the command line arguments.
   *
   * This function parses the command line arguments and stores the values for
   * input path, order, rank, and GPU count based on the provided arguments.
   *
   * @param argc The argument count.
   * @param argv The argument vector.
   * @return A status code indicating the result of parsing.
   */
  ReturnStatus Parse(int argc, char* argv[]);

  /**
   * @brief Retrieves the input path specified in the command line options.
   *
   * @return A constant reference to the input path string. Returns an empty
   * string if no input path is specified.
   */
  const std::string& get_input_path() const;

  /**
   * @brief Retrieves the output path specified in the command line options.
   *
   * @return A constant reference to the output path string. Returns an empty
   * string if no output path is specified.
   */
  const std::string& get_output_path() const;

  /**
   * @brief Retrieves the order of the tensor.
   *
   * @return The order of the tensor.
   */
  inline int get_order() { return this->_order; }

  inline int get_rank() { return this->_rank; }

  inline int get_gpu_count() { return this->_gpu_count; }

  inline int get_host_memory_limit() { return this->_host_memory_limit; }

  inline int get_cuda_stream_count() { return this->_cuda_stream_count; }

  inline bool get_avg_partition() { return this->_avg_partition; }

  /**
   * @brief Initializes the command line options.
   *
   * Sets up the available command line options that the program accepts.
   */
  void Initialize();

  /**
   * @brief Validates the input file specified in the command line options.
   *
   * Checks if the input file exists and is accessible.
   *
   * @return `true` if the file is valid, `false` otherwise.
   */
  bool ValidateFile();

 private:
  po::options_description
      _options;             ///< Describes the available command line options.
  std::string _input_path;  ///< Path to the input file.
  std::string _output_path; ///< Path to the output directory.
  int _order;               ///< Order (rank) of the tensor.
  int _rank;                ///< Rank for the Tucker decomposition.
  int _gpu_count;           ///< Number of GPUs to use for computation.
  int _host_memory_limit;   ///< Memory limit for the host.
  int _cuda_stream_count;   ///< Number of CUDA streams.
  bool _avg_partition;      ///< Flag to enable average-based partitioning.
};  // class CommandLineOptions

/**
 * @brief Retrieves the input path specified in the command line options.
 *
 * This function retrieves the input path specified in the command line options.
 * If no input path is specified, it returns a reference to an empty string.
 *
 * @return A constant reference to the input path string.
 */
inline const std::string& CommandLineOptions::get_input_path() const {
  static const std::string empty_str;
  return (0 < this->_input_path.size() ? this->_input_path : empty_str);
}

inline const std::string& CommandLineOptions::get_output_path() const {
  static const std::string empty_str;
  return (0 < this->_output_path.size() ? this->_output_path : empty_str);
}

}  // namespace gsptucker
}  // namespace supertensor

#endif  // CMDLINE_OPTS_HPP_
