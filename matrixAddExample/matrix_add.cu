#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void matrixAdd1(int *d_A, int *d_B, int *d_C, int sizeX, int sizeY)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int column = blockIdx.x;
    if ((column < sizeX) && (row < sizeY))
        d_C[row * sizeX + column] = d_A[row * sizeX + column] + d_B[row * sizeX + column];
}

double get_time_diff(struct timespec start, struct timespec end)
{
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main(void)
{
    // Define matrix dimensions (rows x columns x depth)
    dim3 dimA(6000, 8000, 1);
    int SizeInBytes_A = sizeof(int) * dimA.x * dimA.y;

    dim3 dimB(6000, 8000, 1);
    int SizeInBytes_B = sizeof(int) * dimB.x * dimB.y;

    dim3 dimC(6000, 8000, 1);
    int SizeInBytes_C = sizeof(int) * dimC.x * dimC.y;

    // Allocate host memory for matrices A and B
    int *h_A = (int *)malloc(SizeInBytes_A);
    int *h_B = (int *)malloc(SizeInBytes_B);
    int *h_C = (int *)malloc(SizeInBytes_C);
    // Initialize the host memory. Notice that srand() and rand() are not thread safe,
    // but we are not concerned with host parallelism at the moment
    srand(time(NULL));
    for (int i = 0; i < dimA.x * dimA.y; i++)
        h_A[i] = rand();
    for (int i = 0; i < dimB.x * dimB.y; i++)
        h_B[i] = rand();

    // select device to execute (device 0) and allocate memory
    int *d_A, *d_B, *d_C;
    if (cudaMalloc(&d_A, SizeInBytes_A) != cudaSuccess)
    {
        printf("CANNOT ALLOCATE d_A\n");
        return -1;
    }
    if (cudaMalloc(&d_B, SizeInBytes_B) != cudaSuccess)
    {
        printf("CANNOT ALLOCATE d_B\n");
        return -1;
    }
    if (cudaMalloc(&d_C, SizeInBytes_C) != cudaSuccess)
    {
        printf("CANNOT ALLOCATE d_C\n");
        return -1;
    }

    // Copy data to the device
    if (cudaMemcpy(d_A, h_A, SizeInBytes_A, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        printf("FAILED TO COPY h_A DATA TO DEVICE\n");
        return -1;
    }
    if (cudaMemcpy(d_B, h_B, SizeInBytes_B, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        printf("FAILED TO COPY h_B DATA TO DEVICE\n");
        return -1;
    }
    // set kernel launch environment:
    // * each thread computes one element
    // * each block computes part of one column (#rows > max number of threads/block)
    int threads_per_block = 256;
    dim3 blockDist(1, threads_per_block, 1);
    dim3 gridDist(dimA.x, (dimA.y + threads_per_block - 1) / threads_per_block, 1);

    // Start timing with high precision
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // launch kernel
    matrixAdd1<<<gridDist, blockDist>>>(d_A, d_B, d_C, dimA.x, dimA.y);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_time = get_time_diff(start, end);

    // get the results
    if (cudaMemcpy(h_C, d_C, SizeInBytes_C, cudaMemcpyDeviceToHost) != cudaSuccess)
        printf("FAILED TO LOAD THE RESULT FROM THE DEVICE");
    else
    {
        printf("Successfully loaded the result from the device\n");
        printf("Time taken: %.6f seconds\n", elapsed_time);
    }

    // clean-up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}