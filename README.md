# GSPTucker

Toward scalable Tucker decomposition: skew-aware multi-level partitioning with GPU-storage co-processing

GSP-Tucker is a scalable and robust Tucker decomposition system designed for billion-scale sparse tensors with severe data skew.
It integrates skew-aware multi-level partitioning, GPU acceleration, and storage-based out-of-core processing to overcome the fundamental scalability limitations of existing Tucker decomposition methods.

Unlike conventional CPU, GPU, and distributed approaches that suffer from intermediate data explosion, memory overflow, and skew-induced workload imbalance, GSP-Tucker provides a unified co-design of partitioning, computation, and I/O scheduling, enabling reliable large-scale tensor decomposition on a single heterogeneous GPU-storage node

## Features

- **GPU Acceleration**: Utilizes NVIDIA GPUs for fast tensor operations using CUDA.
- **Multi-GPU Support**: Scalable across multiple GPUs.
- **Out-of-Core Processing**: Efficiently handles large tensors by utilizing SSDs and host memory, overcoming GPU memory constraints.
- **Multi-level Partitioning**: Implements a multi-level partitioning strategy to effectively handle data skewness and ensure balanced workload distribution across GPUs.

## License

This project is licensed under the terms of the GNU General Public License v3.0 (GPLv3). See the [LICENSE](LICENSE) file for details.

### Disclaimer

**THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW.** EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

## Prerequisites

- **CUDA Toolkit**: (Tested with 11.x)
- **Boost Libraries**: `-lboost_program_options`, `-lboost_filesystem`, `-lboost_system`.
- **Eigen**: Linear algebra library (included in `lib/` or requires installation).
- **OpenMP**: For multi-threading support on the host.

## Build Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tmdgus1715/GSPTucker.git
    cd GSPTucker
    ```

2.  **Configure Makefile:**
    Ensure the `Makefile` points to the correct paths for your Boost installation and CUDA toolkit. You might need to adjust `INCLUDE_DIRS` and `LIB_DIRS`.

3.  **Build:**
    ```bash
    make
    ```
    This will generate the `GSPTucker` executable.

## Usage

Run the executable with the required arguments.

```bash
./GSPTucker -i <input_file> -o <order> [options]
```

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--help` | `-h` | Display help menu. | - |
| `--input` | `-i` | Path to the input tensor file. | **Required** |
| `--output` | `-O` | Path to the output directory (SSD path). | Default (Home) |
| `--order` | `-o` | Order (number of modes) of the tensor. | **Required** |
| `--rank` | `-r` | Tucker rank for the decomposition. | 10 |
| `--gpus` | `-g` | Number of GPUs to use. | 1 |
| `--host_memory_limit` | `-H` | Host memory limit in GB. | 16 |
| `--cuda_stream_count` | `-c` | Number of CUDA streams per GPU. | 1 |
| `--avg_partition` | `-a` | Enable average-based local-skew partitioning for better load balancing. | False |

### Example

```bash
./GSPTucker -i ~/datasets/nell-2.tns -O ./ -o 3 -r 10 -g 1 -c 4 -H 64 -a
```

This command runs Tucker decomposition on `nell-2.tns` (order 3) with rank 10, using 1 GPU, outputting to ./, 4 CUDA streams, a 64GB host memory limit, and average-based partitioning.

## Input Format

The input file should be in a coordinate format, where each line represents a non-zero element:

```
<index_1> <index_2> ... <index_N> <value>

1    1    1    4.0
1    2    1    5.5
2    1    1    3.2
2    2    1    2.8
3    2    1    7.3
1    1    2    1.1
1    2    2    6.8
2    1    2    2.9
2    2    2    4.4
...
```

- Indices are 1-based.
- Indices and value are tap-separated.

Real-world tensor datasets are available in `scripts/datasets.sh`. For more datasets, refer to [FROSTT](http://frostt.io/).

## Directory Structure

- `include/`: Header files (`gsptucker/`, `common/`).
- `source/`: Source files (`gsptucker/`, `common/`).
- `lib/`: External libraries (e.g., Eigen).
- `main.cu`: Main entry point.
- `Makefile`: Build configuration.

