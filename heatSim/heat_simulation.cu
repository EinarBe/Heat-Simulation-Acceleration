#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

template <typename T>
__global__ void heat_diffusion_step(T *T_old, T *T_new, int N, int boundary_row, T alpha1, T alpha2)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < N - 1 && j > 0 && j < N - 1)
    {
        T alpha = (i < boundary_row) ? alpha1 : alpha2;
        T_new[i * N + j] = (1 - alpha) * T_old[i * N + j] +
                           (alpha / 4.0f) * (T_old[(i + 1) * N + j] + T_old[(i - 1) * N + j] +
                                             T_old[i * N + (j + 1)] + T_old[i * N + (j - 1)]);
    }
}

void initialize_grid(float *T, int N, float T_top, float T_other)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            T[i * N + j] = (i == 0) ? T_top : T_other;
        }
    }
}

void save_grid_to_file(float *T, int N, const char *filename)
{
    FILE *file = fopen(filename, "w");
    if (!file)
    {
        perror("Error opening file for writing");
        return;
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            fprintf(file, "%.2f", T[i * N + j]);
            if (j < N - 1)
                fprintf(file, ",");
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

double get_time_diff(struct timespec start, struct timespec end)
{
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main(int argc, char *argv[])
{
    if (argc < 8)
    {
        printf("Usage: %s <grid_size> <boundary_row> <alpha1> <alpha2> <iterations> <T_top> <T_other>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int boundary_row = atoi(argv[2]);
    float alpha1 = atof(argv[3]);
    float alpha2 = atof(argv[4]);
    int iterations = atoi(argv[5]);
    float T_top = atof(argv[6]);
    float T_other = atof(argv[7]);

    float *d_T, *d_T_new, *h_T;
    size_t size = N * N * sizeof(float);
    cudaMalloc(&d_T, size);
    cudaMalloc(&d_T_new, size);
    cudaMallocHost(&h_T, size);

    initialize_grid(h_T, N, T_top, T_other);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    cudaMemcpy(d_T, h_T, size, cudaMemcpyHostToDevice);

    for (int iter = 0; iter < iterations; iter++)
    {
        heat_diffusion_step<<<gridSize, blockSize>>>(d_T, d_T_new, N, boundary_row, alpha1, alpha2);
        cudaDeviceSynchronize();
        float *temp = d_T;
        d_T = d_T_new;
        d_T_new = temp;
    }

    cudaMemcpy(h_T, d_T, size, cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_time = get_time_diff(start, end);

    save_grid_to_file(h_T, N, "heat_output_cuda.csv");

    cudaFree(d_T);
    cudaFree(d_T_new);
    cudaFreeHost(h_T);

    printf("CUDA simulation complete. Results saved to heat_output_cuda.csv\n");
    printf("Calculation loop duration: %.6f seconds\n", elapsed_time);

    return 0;
}