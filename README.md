# Heat Simulation Acceleration

This project implements the acceleration of a 2D heat diffusion simulation using NVIDIA CUDA. By leveraging GPU data-level parallelism and carefully tuned memory access optimizations, the simulation achieves substantial speedups over a traditional CPU implementation.

## Overview

The heat diffusion simulation models how temperature evolves over a 2D grid, where each cell’s new temperature depends on its neighbors. Originally implemented as a CPU-based application in C, the simulation has been restructured and optimized using CUDA to exploit the massive parallelism offered by modern GPUs. Key optimizations include:

- **2D Block Partitioning:** Dividing the simulation grid into two-dimensional blocks (e.g., 16×16) to better align with the data structure.
- **Shared Memory Utilization:** Loading sub-matrices of the grid into shared memory to reduce high-latency global memory accesses.
- **Dual Time-Step Computation:** Computing two simulation steps in a single kernel launch to minimize kernel launch overhead.
- **Multi-GPU Workload Distribution:** Optional support for systems with multiple GPUs, where the workload is partitioned among devices and coordinated with CPU intervention.

These techniques yielded speedups of up to 208× in the best case scenario, as detailed in the accompanying project report.

## Project Structure

```
├── heat_simulation.c   # Original CPU-based implementation of the simulation.
├── heat_simulation.cu  # CUDA-accelerated implementation.
├── Makefile            # Build configuration and targets for compiling, testing, and profiling.
├── visualize_heat.py   # Python script for visualizing simulation outputs.
├── compare_csv.py      # Python script for comparing simulation results (with a tolerance margin).
├── report.pdf          # Project Report.
└── README.md           # This file.
```

## Methodology

The CUDA implementation introduces several key optimizations:

1. **Block Configuration:**  
   The simulation grid is split into 2D blocks (for example, 16×16) to ensure a natural mapping between threads and data points. This minimizes redundant memory accesses by reusing shared boundaries within each block.

2. **Shared Memory Optimization:**  
   Each CUDA kernel copies its assigned grid section and the necessary halo region into shared memory. This drastically reduces the number of expensive global memory accesses during the computation phase.

3. **Dual Time-Step Calculation:**  
   To reduce the overhead of multiple kernel launches, each kernel computes two consecutive simulation steps. Special care is taken to handle boundary conditions properly during these combined computations.

4. **Multi-GPU Support:**  
   On systems with more than one GPU, the grid is partitioned between devices. The CPU manages the overlapping regions to ensure a seamless update of the boundary data.

For a comprehensive discussion of these methodologies and optimizations, please see the attached project report.

## Requirements

- **Hardware:**  
  An NVIDIA GPU with CUDA support (tested on systems with RTX A4000 and GTX 1070).

- **Software:**  
  - CUDA Toolkit (with nvcc, nsys and compute-sanitizer)
  - GCC (or another native compiler; note that native compilers are recommended for proper NEON instruction generation in similar projects)
  - Python 3 (for the visualization and comparison scripts)

## Build and Execution

The project utilizes a Makefile with several targets. Below are some of the most commonly used commands:

### Build Targets

- **Compile the CUDA version:**
  ```bash
  make compile
  ```

- **Run the Simulation:**
  ```bash
  make run
  ```

- **Execute Tests:**  
  To run tests across multiple grid sizes and iteration counts:
  ```bash
  make testmultiplegrid
  make testmultipleiterations
  ```

- **Debug and Profiling:**  
  For memory sanitization and performance analysis:
  ```bash
  make debug    # Runs compute-sanitizer to check memory accesses.
  make profile  # Invokes NVIDIA tools (e.g., nsys) to analyze performance bottlenecks.
  ```

### Python Scripts

Post-simulation, the provided Python scripts can be used for further analysis:

- **Visualization:**
  ```bash
  python3 visualize_heat.py
  ```
- **Result Comparison:**
  ```bash
  python3 compare_csv.py
  ```

## Experimental Results

The accelerated simulation has been evaluated across a range of grid sizes (512 to 8192) and iteration counts (1K to 200K). The results indicate that while the GPU significantly outperforms the CPU across all scenarios, optimal speedups depend on simulation parameters such as grid size, block configuration, and the number of iterations. For example, tests have shown speedups ranging from 25× in challenging conditions up to 208× under optimal configurations.

## Conclusion

By carefully tuning the CUDA kernels—optimizing memory usage, reducing kernel launch overhead, and efficiently partitioning the workload across multiple GPUs—the project demonstrates that significant performance gains are achievable for 2D heat diffusion simulations. The techniques employed here can serve as a reference for similar high-performance computing applications.

## Acknowledgments

This work was developed by Espen van Beuzekom, Einar Bergslid, and João Miranda. For an in-depth discussion of the methodology and experimental results, please refer to the attached project report.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
