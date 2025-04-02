#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define PADDING 2
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void heat_diffusion_2step(float *T_old, float *T_new, int N, int boundary_row, float alpha1, float beta1, float alpha2, float beta2)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = (threadIdx.y + PADDING)*(BLOCK_SIZE_X + 2*PADDING) + threadIdx.x + PADDING;
    extern __shared__ float T_shared [];

    if (i < N && j < N)
        T_shared[tid] = T_old[i * N + j];
    if ((threadIdx.x <= 1 && j > 2) || (threadIdx.x >= BLOCK_SIZE_X - 2 && j < N - 2)) {
        int step = (threadIdx.x <= 1) ? -2 : 2; // TODO: Be careful!
        T_shared[tid + step] = T_old[i * N + (j + step)];
    }
    if ((threadIdx.y <= 1 && i > 2) || (threadIdx.y >= BLOCK_SIZE_Y - 2 && i < N - 2)) {
        int step = (threadIdx.y <= 1) ? -2 : 2; // TODO: Be careful!
        T_shared[tid + step*(BLOCK_SIZE_X + 2*PADDING)] = T_old[(i + step) * N + j];
    }

    __syncthreads();
    float alpha = (i < boundary_row) ? alpha1 : alpha2;
    float beta = (i < boundary_row) ? beta1: beta2;
    float aux [10];
    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        aux[0] = T_shared[tid];
        aux[1] = T_shared[tid - 1];
        aux[2] = T_shared[tid + 1];
        aux[3] = T_shared[tid - (BLOCK_SIZE_X + 2*PADDING)];
        aux[4] = T_shared[tid + (BLOCK_SIZE_X + 2*PADDING)];
        aux[10] = alpha * aux[0] + beta * (aux[1] + aux[2] + aux[3] + aux[4]);
    }
    if (i > 1 && i < N - 2 && j > 1 && j < N - 2) {
        int tid2 = tid - 1;
        aux[6] = (threadIdx.x == 0 && threadIdx.y == 0) ? T_old[(i - 1) * N + (j - 1)] : T_shared[tid2 - (BLOCK_SIZE_X + 2*PADDING)];
        aux[7]= (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE_Y - 1) ? T_old[(i + 1) * N + (j - 1)] : T_shared[tid2 + (BLOCK_SIZE_X + 2*PADDING)];
        aux[1] = alpha * aux[1] + beta * (aux[0] + T_shared[tid2 - 1] + aux[6] + aux[7]);
        tid2 = tid + 1;
        aux[8] = (threadIdx.x == BLOCK_SIZE_X - 1 && threadIdx.y == 0) ? T_old[(i - 1) * N + (j + 1)] : T_shared[tid2 - (BLOCK_SIZE_X + 2*PADDING)];
        aux[9]= (threadIdx.x == BLOCK_SIZE_X - 1 && threadIdx.y == BLOCK_SIZE_Y - 1) ? T_old[(i + 1) * N + (j + 1)] : T_shared[tid2 + (BLOCK_SIZE_X + 2*PADDING)];
        aux[2] = alpha * aux[2] + beta * (T_shared[tid2 + 1] + aux[0] + aux[8] + aux[9]);
        tid2 = tid - (BLOCK_SIZE_X + 2*PADDING);
        aux[3] = alpha * aux[3] + beta * (aux[6] + aux[8] + aux[0] + T_shared[tid2 - (BLOCK_SIZE_X + 2*PADDING)]);
        tid2 = tid + (BLOCK_SIZE_X + 2*PADDING);
        aux[4] = alpha * aux[4] + beta * (aux[7] + aux[9] + T_shared[tid2 + (BLOCK_SIZE_X + 2*PADDING)] + aux[0]);
    }
    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        T_new[i * N + j] = alpha * aux[10] + beta * (aux[1] + aux[2] + aux[3] + aux[4]);
    }
}

__global__ void heat_diffusion_step(float *T_old, float *T_new, int N, int boundary_row, float alpha1, float beta1, float alpha2, float beta2)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = (threadIdx.y + 1)*(blockDim.x + 2) + threadIdx.x + 1;
    extern __shared__ float T_shared [];

    if (i < N && j < N)
        T_shared[tid] = T_old[i * N + j];

    if (threadIdx.x == 0 && j > 0)
        T_shared[tid - 1] = T_old[i * N + (j - 1)];
    if (threadIdx.x == blockDim.x - 1 && j < N - 1)
        T_shared[tid + 1] = T_old[i * N + (j + 1)];
    if (threadIdx.y == 0 && i > 0)
        T_shared[tid - (blockDim.x + 2)] = T_old[(i - 1) * N + j];
    if (threadIdx.y == blockDim.y - 1 && i < N - 1)
        T_shared[tid + (blockDim.x + 2)] = T_old[(i + 1) * N + j];

    __syncthreads();
    if (i > 0 && i < N - 1 && j > 0 && j < N - 1)
    {
        float alpha = (i < boundary_row) ? alpha1 : alpha2;
        float beta = (i < boundary_row) ? beta1: beta2;
        T_new[i * N + j] = alpha * T_shared[tid] +
                           beta * (T_shared[tid + 1] + T_shared[tid - 1] +
                                             T_shared[tid + (blockDim.x + 2)] + T_shared[tid - (blockDim.x + 2)]);
    }
}

__global__ void initialize_grid_kernel(float *T, int N, float T_top, float T_other) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N)
        T[i * N + j] = (i == 0) ? T_top : T_other;
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
    float cte1 = atof(argv[3]);
    float cte2 = atof(argv[4]);
    int iterations = atoi(argv[5]);
    float T_top = atof(argv[6]);
    float T_other = atof(argv[7]);

    float alpha1 = (1 - cte1);
    float alpha2 = (1 - cte2);
    float beta1 = cte1/4.0f;
    float beta2 = cte2/4.0f;

    float *d_T, *d_T_new, *h_T;
    size_t size = N * N * sizeof(float);
    cudaMalloc(&d_T, size);
    cudaMalloc(&d_T_new, size);
    cudaMallocHost(&h_T, size);

    initialize_grid(h_T, N, T_top, T_other);

    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    cudaMemcpy(d_T_new, h_T, size, cudaMemcpyHostToDevice); // TODO: Check
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    initialize_grid_kernel<<<gridSize, blockSize>>>(d_T, N, T_top, T_other);

    int iterations_mod = (iterations) - (iterations % 2);
    int iter;
    for (iter = 0; iter < iterations_mod; iter+=2) {
        heat_diffusion_2step<<<gridSize, blockSize, (BLOCK_SIZE_X + 2*PADDING)*(BLOCK_SIZE_Y + 2*PADDING)*(sizeof(float))>>>(d_T, d_T_new, N, boundary_row, alpha1, beta1, alpha2, beta2);
        cudaDeviceSynchronize();
        float *temp = d_T;
        d_T = d_T_new;
        d_T_new = temp;
    }
    while(iter < iterations) {
        heat_diffusion_step<<<gridSize, blockSize, (BLOCK_SIZE_X + 2*PADDING)*(BLOCK_SIZE_Y + 2*PADDING)*(sizeof(float))>>>(d_T, d_T_new, N, boundary_row, alpha1, beta1, alpha2, beta2);
        cudaDeviceSynchronize();
        float *temp = d_T;
        d_T = d_T_new;
        d_T_new = temp;
        iter ++;
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