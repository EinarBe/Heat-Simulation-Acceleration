#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define PADDING 2
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define CPU_LINES_DIV_8 1
#define CPU_LINES 8*CPU_LINES_DIV_8

__global__ void heat_diffusion_2step(float *T_old, float *T_new, int N, int boundary_row, float alpha1, float beta1, float alpha2, float beta2)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = (threadIdx.y + PADDING)*(BLOCK_SIZE_X + 2*PADDING) + threadIdx.x + PADDING;
    extern __shared__ float T_shared [];

    if (i < N && j < N)
        T_shared[tid] = T_old[i * N + j];
    if ((threadIdx.x <= 1 && j > 2) || (threadIdx.x >= BLOCK_SIZE_X - 2 && j < N - 2)) {
        int step = (threadIdx.x <= 1) ? -2 : 2;
        T_shared[tid + step] = T_old[i * N + (j + step)];
    }
    if ((threadIdx.y <= 1 && i > 2) || (threadIdx.y >= BLOCK_SIZE_Y - 2 && i < N - 2)) {
        int step = (threadIdx.y <= 1) ? -2 : 2;
        T_shared[tid + step*(BLOCK_SIZE_X + 2*PADDING)] = T_old[(i + step) * N + j];
    }

    __syncthreads();
    float alpha = (i < boundary_row) ? alpha1 : alpha2;
    float beta = (i < boundary_row) ? beta1: beta2;
    float aux [6];
    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        aux[0] = T_shared[tid];
        aux[1] = T_shared[tid - 1];
        aux[2] = T_shared[tid + 1];
        aux[3] = T_shared[tid - (BLOCK_SIZE_X + 2*PADDING)];
        aux[4] = T_shared[tid + (BLOCK_SIZE_X + 2*PADDING)];
        aux[6] = 2 * alpha * (aux[1] + aux[2] + aux[3] + aux[4]);
        aux[5] = 4 * aux[0];
        int tid2 = tid - (BLOCK_SIZE_X + 2*PADDING);
        aux[1] = (threadIdx.x == 0 && threadIdx.y == 0) ? T_old[(i - 1) * N + (j - 1)] : T_shared[tid2 - 1];
        aux[2] = (threadIdx.x == BLOCK_SIZE_X - 1 && threadIdx.y == 0) ? T_old[(i - 1) * N + (j + 1)] : T_shared[tid2 + 1];
        aux[5] += 2 * (aux[1] + aux[2]);
        aux[5] += (j > 1) ? T_shared[tid - 2] : - aux[0];
        tid2 = tid + (BLOCK_SIZE_X + 2*PADDING);
        aux[3] = (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE_Y - 1) ? T_old[(i + 1) * N + (j - 1)] : T_shared[tid2 - 1];
        aux[4] = (threadIdx.x == BLOCK_SIZE_X - 1 && threadIdx.y == BLOCK_SIZE_Y - 1) ? T_old[(i + 1) * N + (j + 1)] : T_shared[tid2 + 1];
        aux[5] += 2 * (aux[3] + aux[4]);
        aux[5] += (j < N - 2) ? T_shared[tid + 2] : - aux[0];
        aux[5] += (i > 1) ? T_shared[tid - 2*(BLOCK_SIZE_X + 2*PADDING)] : - aux[0];
        aux[5] += (i <  N - 2) ? T_shared[tid + 2*(BLOCK_SIZE_X + 2*PADDING)] : - aux[0];
        T_new[i * N + j] = alpha * alpha * aux[0] + beta * (aux[6] + beta * aux[5]);
    }
}

__global__ void heat_diffusion_2step_d0(float *T_old, float *T_new, int N, int boundary_row, float alpha1, float beta1, float alpha2, float beta2)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = (threadIdx.y + PADDING)*(BLOCK_SIZE_X + 2*PADDING) + threadIdx.x + PADDING;
    extern __shared__ float T_shared [];

    if (i < (N >> 1) && j < N)
        T_shared[tid] = T_old[i * N + j];
    if ((threadIdx.x <= 1 && j > 2) || (threadIdx.x >= BLOCK_SIZE_X - 2 && j < N - 2)) {
        int step = (threadIdx.x <= 1) ? -2 : 2;
        T_shared[tid + step] = T_old[i * N + (j + step)];
    }
    if ((threadIdx.y <= 1 && i > 2) || (threadIdx.y >= BLOCK_SIZE_Y - 2 && i < (N >> 1) - 2)) {
        int step = (threadIdx.y <= 1) ? -2 : 2;
        T_shared[tid + step*(BLOCK_SIZE_X + 2*PADDING)] = T_old[(i + step) * N + j];
    }

    __syncthreads();
    float alpha = (i < boundary_row) ? alpha1 : alpha2;
    float beta = (i < boundary_row) ? beta1: beta2;
    float aux [6];
    if (i > 0 && i < (N >> 1) - 1 && j > 0 && j < N - 1) {
        aux[0] = T_shared[tid];
        aux[1] = T_shared[tid - 1];
        aux[2] = T_shared[tid + 1];
        aux[3] = T_shared[tid - (BLOCK_SIZE_X + 2*PADDING)];
        aux[4] = T_shared[tid + (BLOCK_SIZE_X + 2*PADDING)];
        aux[6] = 2 * alpha * (aux[1] + aux[2] + aux[3] + aux[4]);
        aux[5] = 4 * aux[0];
        int tid2 = tid - (BLOCK_SIZE_X + 2*PADDING);
        aux[1] = (threadIdx.x == 0 && threadIdx.y == 0) ? T_old[(i - 1) * N + (j - 1)] : T_shared[tid2 - 1];
        aux[2] = (threadIdx.x == BLOCK_SIZE_X - 1 && threadIdx.y == 0) ? T_old[(i - 1) * N + (j + 1)] : T_shared[tid2 + 1];
        aux[5] += 2 * (aux[1] + aux[2]);
        aux[5] += (j > 1) ? T_shared[tid - 2] : - aux[0];
        tid2 = tid + (BLOCK_SIZE_X + 2*PADDING);
        aux[3] = (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE_Y - 1) ? T_old[(i + 1) * N + (j - 1)] : T_shared[tid2 - 1];
        aux[4] = (threadIdx.x == BLOCK_SIZE_X - 1 && threadIdx.y == BLOCK_SIZE_Y - 1) ? T_old[(i + 1) * N + (j + 1)] : T_shared[tid2 + 1];
        aux[5] += 2 * (aux[3] + aux[4]);
        aux[5] += (j < N - 2) ? T_shared[tid + 2] : - aux[0];
        aux[5] += (i > 1) ? T_shared[tid - 2*(BLOCK_SIZE_X + 2*PADDING)] : - aux[0];
        aux[5] += (i <  (N >> 1) - 2) ? T_shared[tid + 2*(BLOCK_SIZE_X + 2*PADDING)] : - aux[0];
        T_new[i * N + j] = alpha * alpha * aux[0] + beta * (aux[6] + beta * aux[5]);
    }
}

__global__ void heat_diffusion_2step_d1(float *T_old, float *T_new, int N, int boundary_row, float alpha1, float beta1, float alpha2, float beta2)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = (threadIdx.y + PADDING)*(BLOCK_SIZE_X + 2*PADDING) + threadIdx.x + PADDING;
    extern __shared__ float T_shared [];

    if (i < (N >> 1) && j < N)
        T_shared[tid] = T_old[i * N + j];
    if ((threadIdx.x <= 1 && j > 2) || (threadIdx.x >= BLOCK_SIZE_X - 2 && j < N - 2)) {
        int step = (threadIdx.x <= 1) ? -2 : 2;
        T_shared[tid + step] = T_old[i * N + (j + step)];
    }
    if ((threadIdx.y <= 1 && i > 2) || (threadIdx.y >= BLOCK_SIZE_Y - 2 && i < (N >> 1) - 2)) {
        int step = (threadIdx.y <= 1) ? -2 : 2;
        T_shared[tid + step*(BLOCK_SIZE_X + 2*PADDING)] = T_old[(i + step) * N + j];
    }

    __syncthreads();
    float alpha = (i < boundary_row) ? alpha1 : alpha2;
    float beta = (i < boundary_row) ? beta1: beta2;
    float aux [6];
    if (i > 0 && i < (N >> 1) - 1 && j > 0 && j < N - 1) {
        aux[0] = T_shared[tid];
        aux[1] = T_shared[tid - 1];
        aux[2] = T_shared[tid + 1];
        aux[3] = T_shared[tid - (BLOCK_SIZE_X + 2*PADDING)];
        aux[4] = T_shared[tid + (BLOCK_SIZE_X + 2*PADDING)];
        aux[6] = 2 * alpha * (aux[1] + aux[2] + aux[3] + aux[4]);
        aux[5] = 4 * aux[0];
        int tid2 = tid - (BLOCK_SIZE_X + 2*PADDING);
        aux[1] = (threadIdx.x == 0 && threadIdx.y == 0) ? T_old[(i - 1) * N + (j - 1)] : T_shared[tid2 - 1];
        aux[2] = (threadIdx.x == BLOCK_SIZE_X - 1 && threadIdx.y == 0) ? T_old[(i - 1) * N + (j + 1)] : T_shared[tid2 + 1];
        aux[5] += 2 * (aux[1] + aux[2]);
        aux[5] += (j > 1) ? T_shared[tid - 2] : - aux[0];
        tid2 = tid + (BLOCK_SIZE_X + 2*PADDING);
        aux[3] = (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE_Y - 1) ? T_old[(i + 1) * N + (j - 1)] : T_shared[tid2 - 1];
        aux[4] = (threadIdx.x == BLOCK_SIZE_X - 1 && threadIdx.y == BLOCK_SIZE_Y - 1) ? T_old[(i + 1) * N + (j + 1)] : T_shared[tid2 + 1];
        aux[5] += 2 * (aux[3] + aux[4]);
        aux[5] += (j < N - 2) ? T_shared[tid + 2] : - aux[0];
        aux[5] += (i > 1) ? T_shared[tid - 2*(BLOCK_SIZE_X + 2*PADDING)] : - aux[0];
        aux[5] += (i <  (N >> 1) - 2) ? T_shared[tid + 2*(BLOCK_SIZE_X + 2*PADDING)] : - aux[0];
        T_new[i * N + j] = alpha * alpha * aux[0] + beta * (aux[6] + beta * aux[5]);
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

__global__ void heat_diffusion_step_d0(float *T_old, float *T_new, int N, int boundary_row, float alpha1, float beta1, float alpha2, float beta2)
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

__global__ void heat_diffusion_step_d1(float *T_old, float *T_new, int N, int boundary_row, float alpha1, float beta1, float alpha2, float beta2)
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

__global__ void initialize_grid_kernel(float *T, int N, float T_top, float T_other, int device) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (N >> 1) && j < N)
        T[i * N + j] = (i == 0 && device == 0) ? T_top : T_other;
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

void initialize_grid2(float *T, int N, float T_top, float T_other)
{
    for (int i = 0; i < CPU_LINES; i++)
    {
        for (int j = 0; j < N; j++)
        {
            T[i * N + j] = T_other;
        }
    }
}

void complete_grid(float *T, const float* T2, int N)
{
    for (int i = 0; i < 4*CPU_LINES_DIV_8; i++)
    {
        for (int j = 0; j < N; j++)
        {
            T[((N >> 1) - 2*CPU_LINES_DIV_8 + i) * N + j] = T2[i * N + j];
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

void heat_diffusion_step_h(float *T, float *T_new, int N, int boundary_row,
                         float alpha1, float beta1,  float alpha2, float beta2) {
#pragma omp parallel for
  for (int i = 1; i < CPU_LINES - 1; i++) {
#pragma omp simd
    for (int j = 1; j < N - 1; j++) {
      float alpha = (i < boundary_row) ? alpha1 : alpha2;
      float beta = (i < boundary_row) ? beta1 : beta2;
      T_new[i * N + j] =
          alpha * T[i * N + j] +
          beta * (T[(i + 1) * N + j] + T[(i - 1) * N + j] +
                            T[i * N + (j + 1)] + T[i * N + (j - 1)]);
    }
  }
}

void run_on_1gpu(int N, int boundary_row, float cte1, float cte2, int iterations, float T_top, float T_other) {
    float *d_T, *d_T_new, *h_T;
    size_t size = N * N * sizeof(float);
    cudaMalloc(&d_T, size);
    cudaMalloc(&d_T_new, size);
    cudaMallocHost(&h_T, size);

    printf("1 GPU\n");

    initialize_grid(h_T, N, T_top, T_other);

    float alpha1 = (1 - cte1);
    float alpha2 = (1 - cte2);
    float beta1 = cte1/4.0f;
    float beta2 = cte2/4.0f;

    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    cudaMemcpy(d_T_new, h_T, size, cudaMemcpyHostToDevice);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    initialize_grid_kernel<<<gridSize, blockSize, 0, stream>>>(d_T, N, T_top, T_other, 0);

    int iterations_mod = (iterations) - (iterations % 2);
    int iter;
    for (iter = 0; iter < iterations_mod; iter+=2) {
        heat_diffusion_2step<<<gridSize, blockSize, (BLOCK_SIZE_X + 2*PADDING)*(BLOCK_SIZE_Y + 2*PADDING)*(sizeof(float)), stream>>>(d_T, d_T_new, N, boundary_row, alpha1, beta1, alpha2, beta2);
        cudaStreamSynchronize(stream);
        float *temp = d_T;
        d_T = d_T_new;
        d_T_new = temp;
    }
    while(iter < iterations) {
        heat_diffusion_step<<<gridSize, blockSize, (BLOCK_SIZE_X + 2*PADDING)*(BLOCK_SIZE_Y + 2*PADDING)*(sizeof(float)), stream>>>(d_T, d_T_new, N, boundary_row, alpha1, beta1, alpha2, beta2);
        cudaStreamSynchronize(stream);
        float *temp = d_T;
        d_T = d_T_new;
        d_T_new = temp;
        iter ++;
    }

    cudaMemcpyAsync(h_T, d_T, size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_time = get_time_diff(start, end);

    save_grid_to_file(h_T, N, "heat_output_cuda.csv");

    cudaFree(d_T);
    cudaFree(d_T_new);
    cudaFreeHost(h_T);

    printf("CUDA simulation complete. Results saved to heat_output_cuda.csv\n");
    printf("Calculation loop duration: %.6f seconds\n", elapsed_time);
}

void run_on_2gpus(int N, int boundary_row, float cte1, float cte2, int iterations, float T_top, float T_other) {
    float *d0_T, *d0_T_new, *d1_T, *d1_T_new, *h_T, *h_T_new, *T;
    size_t size = N * (N/2) * sizeof(float);
    size_t size_host = CPU_LINES * N * sizeof(float);
    size_t transfer_size = 2*CPU_LINES_DIV_8 * N * sizeof(float);

    printf("2 GPUs\n");

    cudaMallocHost(&h_T, size_host);
    cudaMallocHost(&h_T_new, size_host);
    cudaMallocHost(&T, N * N * sizeof(float));

    initialize_grid2(h_T, N, T_top, T_other);
    initialize_grid2(h_T_new, N, T_top, T_other);

    float alpha1 = (1 - cte1);
    float alpha2 = (1 - cte2);
    float beta1 = cte1/4.0f;
    float beta2 = cte2/4.0f;

    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, ((N >> 1) + blockSize.y - 1) / blockSize.y);

    cudaSetDevice(0);
    cudaMalloc(&d0_T, size);
    cudaMalloc(&d0_T_new, size);

    cudaStream_t stream0;
    cudaStreamCreate(&stream0);
    cudaStream_t stream2;
    cudaStreamCreate(&stream2);

    initialize_grid_kernel<<<gridSize, blockSize>>>(d0_T_new, N, T_top, T_other, 0);

    cudaSetDevice(1);
    cudaMalloc(&d1_T, size);
    cudaMalloc(&d1_T_new, size);

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStream_t stream3;
    cudaStreamCreate(&stream3);

    initialize_grid_kernel<<<gridSize, blockSize>>>(d1_T_new, N, T_top, T_other, 1);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    cudaSetDevice(0);
    initialize_grid_kernel<<<gridSize, blockSize, 0, stream0>>>(d0_T, N, T_top, T_other, 0);
    
    cudaSetDevice(1);
    initialize_grid_kernel<<<gridSize, blockSize, 0, stream1>>>(d1_T, N, T_top, T_other, 1);

    int iterations_mod = (iterations) - (iterations % 2);
    int iter, k = 0;
    for (iter = 0; iter < iterations_mod; iter+=2) {
        cudaSetDevice(0);
        heat_diffusion_2step_d0<<<gridSize, blockSize, (BLOCK_SIZE_X + 2*PADDING)*(BLOCK_SIZE_Y + 2*PADDING)*(sizeof(float)), stream0>>>(d0_T, d0_T_new, N, boundary_row, alpha1, beta1, alpha2, beta2);
        cudaSetDevice(1);
        heat_diffusion_2step_d1<<<gridSize, blockSize, (BLOCK_SIZE_X + 2*PADDING)*(BLOCK_SIZE_Y + 2*PADDING)*(sizeof(float)), stream1>>>(d1_T, d1_T_new, N, boundary_row - (N >> 1), alpha1, beta1, alpha2, beta2);
        heat_diffusion_step_h(h_T, h_T_new, N, boundary_row - (N >> 1) + (CPU_LINES >> 1), alpha1, beta1, alpha2, beta2);
        float *temp = h_T;
        h_T = h_T_new;
        h_T_new = temp;
        cudaSetDevice(0);
        cudaStreamSynchronize(stream0);
        temp = d0_T;
        d0_T = d0_T_new;
        d0_T_new = temp;
        k ++;
        int copy = ((k % CPU_LINES_DIV_8) == 0);
        if(copy) {
            cudaMemcpyAsync(h_T, &d0_T[((N >> 1) - 4*CPU_LINES_DIV_8)*N], transfer_size, cudaMemcpyDeviceToHost, stream0);
            cudaMemcpyAsync(&d0_T[((N >> 1) - 2*CPU_LINES_DIV_8)*N], &h_T[2*CPU_LINES_DIV_8*N], transfer_size, cudaMemcpyHostToDevice, stream2);
        }
        cudaSetDevice(1);
        cudaStreamSynchronize(stream1);
        temp = d1_T;
        d1_T = d1_T_new;
        d1_T_new = temp;
        if(copy) {
            cudaMemcpyAsync(&h_T[6*CPU_LINES_DIV_8*N], &d1_T[(2*CPU_LINES_DIV_8)*N], transfer_size, cudaMemcpyDeviceToHost, stream1);
            cudaMemcpyAsync(d1_T, &h_T[4*CPU_LINES_DIV_8*N], transfer_size, cudaMemcpyHostToDevice, stream3);
            cudaStreamSynchronize(stream0);
            cudaStreamSynchronize(stream2);
            cudaStreamSynchronize(stream1);
            cudaStreamSynchronize(stream3);
        }
    }
    while(iter < iterations) {
        cudaSetDevice(0);
        heat_diffusion_step_d0<<<gridSize, blockSize, (BLOCK_SIZE_X + 2*PADDING)*(BLOCK_SIZE_Y + 2*PADDING)*(sizeof(float)), stream0>>>(d0_T, d0_T_new, N, boundary_row, alpha1, beta1, alpha2, beta2);
        cudaSetDevice(1);
        heat_diffusion_step_d1<<<gridSize, blockSize, (BLOCK_SIZE_X + 2*PADDING)*(BLOCK_SIZE_Y + 2*PADDING)*(sizeof(float)), stream1>>>(d1_T, d1_T_new, N, boundary_row - (N >> 1), alpha1, beta1, alpha2, beta2);
        heat_diffusion_step_h(h_T, h_T_new, N, boundary_row - (N >> 1) + (CPU_LINES >> 1), alpha1, beta1, alpha2, beta2);
        float *temp = h_T;
        h_T = h_T_new;
        h_T_new = temp;
        cudaSetDevice(0);
        cudaStreamSynchronize(stream0);
        temp = d0_T;
        d0_T = d0_T_new;
        d0_T_new = temp;
        cudaSetDevice(1);
        cudaStreamSynchronize(stream1);
        temp = d1_T;
        d1_T = d1_T_new;
        d1_T_new = temp;
        iter ++;
    }

    cudaMemcpyAsync(T, d0_T, N * ((N >> 1) - 2*CPU_LINES_DIV_8) * sizeof(float), cudaMemcpyDeviceToHost, stream0);
    cudaMemcpyAsync(&T[((N >> 1) + 2*CPU_LINES_DIV_8) * N], d1_T, N * ((N >> 1) - 2*CPU_LINES_DIV_8) * sizeof(float), cudaMemcpyDeviceToHost, stream1);
    complete_grid(T, h_T, N);
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_time = get_time_diff(start, end);

    cudaSetDevice(0);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream2);
    cudaFree(d0_T);
    cudaFree(d0_T_new);

    cudaSetDevice(1);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream3);
    cudaFree(d1_T);
    cudaFree(d1_T_new);

    save_grid_to_file(T, N, "heat_output_cuda.csv");

    cudaFreeHost(h_T);
    cudaFreeHost(h_T_new);
    cudaFreeHost(T);

    printf("CUDA simulation complete. Results saved to heat_output_cuda.csv\n");
    printf("Calculation loop duration: %.6f seconds\n", elapsed_time);
}

int main(int argc, char *argv[]) {
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

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if(deviceCount == 1) {
        run_on_1gpu(N, boundary_row, cte1, cte2, iterations, T_top, T_other);
    } else if(deviceCount == 2) {
        run_on_2gpus(N, boundary_row, cte1, cte2, iterations, T_top, T_other);
    } else {
        printf("ERROR: No code for more than 2 GPUs");
    }

    return 0;
}
