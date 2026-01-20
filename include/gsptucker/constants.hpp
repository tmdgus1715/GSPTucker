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

#ifndef CONSTANTS_HPP_
#define CONSTANTS_HPP_

namespace supertensor {
namespace gsptucker {

/**
 * @brief Contains constant values used in the Tucker decomposition algorithm.
 *
 * This namespace defines a set of constant values that are used throughout
 * the Tucker decomposition implementation. These constants are used to configure
 * various aspects of the algorithm, such as the maximum order of tensors, the
 * maximum number of iterations, and regularization parameters.
 */
namespace constants {
constexpr int kMaxOrder{8};        ///< Maximum order (rank) of tensors supported.
constexpr int kMaxIteration{1};    ///< Maximum number of iterations for the algorithm.
constexpr double kLambda{0.001f}; ///< Regularization parameter used in the decomposition.
} // namespace constants

/**
 * @brief Enumerations used in the Tucker decomposition algorithm.
 *
 * This namespace contains enumerations that define various types of partitions
 * used during the Tucker decomposition process. The partitions determine how
 * the tensor data is divided for processing, especially when utilizing CUDA
 * streaming for large-scale tensors.
 */
namespace enums {

/**
 * @brief Enumeration of partition types used in the decomposition.
 *
 * These enumeration values indicate different strategies for partitioning the
 * tensor data. The chosen partitioning method can affect the efficiency and
 * performance of the Tucker decomposition, particularly in GPU-accelerated environments.
 */
enum PartitionTypes {
  kDimensionPartition, ///< Partitioning based on tensor dimensions, suitable for large-scale tensors with CUDA streaming.
  kNonzeroPartition,   ///< Partitioning based on non-zero elements, suitable for small-scale tensors without CUDA streaming.
  kPartitionTypeCount  ///< The total count of partition types available.
};

} // namespace enums

} // namespace gsptucker
} // namespace supertensor

#endif /* CONSTANTS_HPP_ */
