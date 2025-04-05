# General options
CC = gcc
NVCC = nvcc
CFLAGS ?= -O3 -lm -fopenmp # Compiler options
NVFLAGS ?= -O3 -Xptxas=-v # Compiler options

# Default target
TARGET = heat_simulation
SRC = heat_simulation.c
OBJ = $(SRC:.c=.o)

# Cuda target
CUDA_TARGET = heat_simulation_cuda
CUDA_SRC = heat_simulation.cu

# Default simulation options
GRID_SIZE ?= 1024
BOUNDARY_ROW ?= 512
ALPHA1 ?= 0.2
ALPHA2 ?= 0.8
ITERATIONS ?= 10000
T_TOP ?= 100.0
T_OTHER ?= 0.0

# Vizualization file
HEATFILE = heat_output.csv

# Compile the program
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ)

$(OBJ): $(SRC)
	$(CC) $(CFLAGS) -c $(SRC)

cuda:
	$(NVCC) $(NVFLAGS) $(CUDA_SRC) -o $(CUDA_TARGET)
	@./$(CUDA_TARGET) $(GRID_SIZE) $(BOUNDARY_ROW) $(ALPHA1) $(ALPHA2) $(ITERATIONS) $(T_TOP) $(T_OTHER)
	python3 compare_csv.py

debug:
	$(NVCC) $(NVFLAGS) $(CUDA_SRC) -o $(CUDA_TARGET)
	compute-sanitizer --tool memcheck ./$(CUDA_TARGET) $(GRID_SIZE) $(BOUNDARY_ROW) $(ALPHA1) $(ALPHA2) $(ITERATIONS) $(T_TOP) $(T_OTHER)

# Clean up binary files
clean:
	rm -f $(TARGET) $(OBJ) $(CUDA_TARGET) heat_output.csv heatmap.png
	
# Run the simulation
run:
	@./$(TARGET) $(GRID_SIZE) $(BOUNDARY_ROW) $(ALPHA1) $(ALPHA2) $(ITERATIONS) $(T_TOP) $(T_OTHER)

profile:
	nsys profile ./$(CUDA_TARGET) $(GRID_SIZE) $(BOUNDARY_ROW) $(ALPHA1) $(ALPHA2) $(ITERATIONS) $(T_TOP) $(T_OTHER)

# Run heatmap visualization
visualize:
	@if [ "$(wildcard $(HEATFILE))" ]; then \
		python3 visualize_heat.py; \
	else \
		echo "Output file '$(HEATFILE)' does not exist"; \
		echo "Please type 'make run' to generate the file"; \
	fi

# Show usage instructions and target details
help:
	@echo "Usage: make [target] [CFLAGS=\"flags\"]"
	@echo ""
	@echo "Targets:"
	@echo ""
	@echo "  all         - Compile the simulation file for heat diffusion (default option)"
	@echo ""
	@echo "  run         - Run the simulation with default parameters"
	@echo "                Options for run are:"
	@echo "                  * GRID_SIZE=grid size (default: 1024) "
	@echo "                  * BOUNDARY_ROW=row (default: 512, should be in range [1,GRID_SIZE]) "
	@echo "                  * ALPHA1=thermal diffusion of material 1 (default: 0.2, should be "
	@echo "                           in range [0.0 ; 1.0]) "
	@echo "                  * ALPHA2=thermal diffusion of material 2, i.e., after boundary row "
	@echo "                           (default: 0.8, should be in range [0.0 ; 1.0]) "
	@echo "                  * ITERATIONS=number of iterations to perform (default: 10000)"
	@echo "                  * T_TOP=temperature at the top row (default: 100.0)"
	@echo "                  * T_OTHER=initial temperature of the other rows (default: 0.0)"
	@echo "                example:"
	@echo "                  make run GRID_SIZE=2048 BOUNDARY_ROW=100 ALPHA1=0.3 ALPHA2=0.5 \\"
	@echo "                           ITERATIONS=1000000 T_TOP=80 T_OTHER=10"
	@echo ""
	@echo "  visualize   - Show the heatmap"
	@echo ""
	@echo "  clean       - Remove compiled files and output data"
	@echo ""
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make CFLAGS=\"-O3 -lm\"              # Compile with level 3 optimizations"
	@echo "  make CFLAGS=\"-O3 -lm -fopenmp\"     # Compile with level 3 optimizations and OpenMP"
	@echo "  make run                           # Run with defaults"
	@echo "  make run ITERATIONS=100            # Run with only 100 iterations"
	@echo "  make visualize                     # Run and generate heatmap"

# Phony targets to avoid conflicts with filenames
.PHONY: all clean visualize run help
