#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

// Tiled shared memory convolution
#define TILE_WIDTH 8

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    
    int W_grid = ceil(1.0*W_out / TILE_WIDTH);
    // int H_grid = H_out / TILE_WIDTH;
    // int Z = H_grid * W_grid;
    
    int x_tile_width = TILE_WIDTH + K - 1;
    extern __shared__ float shmem[];
    float *x_shared = &shmem[0];
    float *w_shared = &shmem[x_tile_width * x_tile_width];
    
    int b = blockIdx.x;
    int m = blockIdx.y;
    int w0 = threadIdx.y;
    int h0 = threadIdx.x;
    int w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    int h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    int w = w_base + w0;
    int h = h_base + h0;

    float acc = 0;
    
    for (int c = 0; c < C; c++){
        for (int i = h0; i < K; i += TILE_WIDTH){
            for (int j = w0; j < K; j += TILE_WIDTH){
                w_shared[i * K + j] = k4d(m, c, i, j);
            }
        }

        __syncthreads();

        for (int i = h; i < h_base + x_tile_width; i += TILE_WIDTH){
            for (int j = w; j < w_base + x_tile_width; j += TILE_WIDTH){
                if (i < H && j < W){
                    x_shared[(i - h_base) * x_tile_width + (j - w_base)] = x4d(b, c, i, j);
                }
            }
        }

        __syncthreads();

        for (int p = 0; p < K; ++p){
            for (int q = 0; q < K; ++q){
                //!!!!x_shared
                acc += x_shared[(h0 + p) * x_tile_width + (w0 + q)] * w_shared[p * K + q];
            }
        }

        __syncthreads();

        if (h < H_out && w < W_out){
            y4d(b, m, h, w) = acc;    
        }

    }


#undef y4d
#undef x4d
}

// Shared memory matrix multiplication and input matrix unrolling

// #define TILE_WIDTH 16
// #define BLOCK_SIZE 1024

__global__ void matrix_multiply_shared(const float *A, float *B, float *C, int numAColumns, int numCRows, int numCColumns)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float pvalue = 0;

    for (int m = 0 ; m < ceil(1.0 * numAColumns / TILE_WIDTH); m++){
        if (Row < numCRows && (m * TILE_WIDTH + tx) < numAColumns){
            tileA[ty][tx] = A[Row * numAColumns + m * TILE_WIDTH + tx];
        }
        else {
            tileA[ty][tx] = 0;
        }

        if (Col < numCColumns && (m * TILE_WIDTH + ty) < numAColumns){
            tileB[ty][tx] = B[(m * TILE_WIDTH + ty) * numCColumns + Col];
        }
        else{
            tileB[ty][tx] = 0;
        }

        __syncthreads();
            
        
        if (Row < numCRows && Col < numCColumns){
            for (int k = 0; k < TILE_WIDTH; k++){
               pvalue += tileA[ty][k] * tileB[k][tx];
           }
        }
        
        __syncthreads();   

    }

    if (Row < numCRows && Col < numCColumns){
        C[Row * numCColumns + Col] = pvalue;
    }

}

__global__ void unroll_kernel(const float *x, float *x_unroll, const int C, const int H, const int W, const int K)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

    if (t < C * W_unroll){
        int c = t / W_unroll;
        int s = t % W_unroll;
        int h_out = s / W_out;
        int w_out = s % W_out;
        // !!!
        int w_unroll = h_out * W_out + w_out;
        int w_base = c * K * K;

        for (int p = 0; p < K; p++){
            for (int q = 0; q < K; q++){
                // !!!
                int h_unroll = w_base + p * K + q;
                x_unroll[h_unroll * W_unroll + w_unroll] = x[c * H * W + (h_out + p) * W + w_out + q];
            }
        }

    }
}

// Kernel fusion for unrolling and matrix-multiplication and 
// Weight matrix (kernel values) in constant memory

// #define TILE_WIDTH 16
__constant__ float k[7200];

__global__ void fusion_kernel(const float *x, float *y, const int C, const int K, const int H, const int W, const int M)
{

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
   
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int b = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
 
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int H_unroll = C * K * K;
    int W_unroll = H_out * W_out;

    float pvalue = 0.0;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    for (int m = 0 ; m < ceil(1.0 * H_unroll / TILE_WIDTH); m++){
        
        int row_tile = m * TILE_WIDTH + ty;
        int col_tile = m * TILE_WIDTH + tx;

        int k_m = row;
        int k_c = col_tile / (K*K);
        int k_k1 = (col_tile % (K*K)) / K;
        int k_k2 = (col_tile % (K*K)) % K;

        if (row < M && col_tile < H_unroll){
            tileA[ty][tx] = k4d(k_m, k_c, k_k1, k_k2);
        }
        else {
            tileA[ty][tx] = 0.0;
        }

        int x_b = b;
        int x_c = row_tile / (K*K);
        int x_h = (row_tile % (K*K)) / K;
        int x_w = (row_tile % (K*K)) % K;
        
        if (col < W_unroll && row_tile < H_unroll){
            tileB[ty][tx] = x4d(x_b, x_c, x_h + col/W_out, x_w + col%W_out);
        }
        else{
            tileB[ty][tx] = 0.0;
        }

        __syncthreads();
    
        for (int n = 0; n < TILE_WIDTH; n++){
           pvalue += tileA[ty][n] * tileB[n][tx];
       }
        
        __syncthreads();   

    }

    if (row < M && col < W_unroll){
        y4d(b, row, col/W_out, col%W_out) = pvalue;
    }
    __syncthreads();

#undef y4d
#undef x4d
#undef k4d

}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
    float *device_x;
    float *device_y;
    float *device_k;

    int numOutputElements = B * M * H_out * W_out;
    int numInputElements  = B * C * H * W;
    int kernelLength      = M * C * K * K;
    
    cudaMalloc((void **) &device_x,  numInputElements * sizeof(float));
    cudaMalloc((void **) &device_y,  numOutputElements * sizeof(float));
    cudaMalloc((void **) &device_k,  kernelLength * sizeof(float));
   
    cudaMemcpy(device_x, host_x, numInputElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_k, host_k, kernelLength * sizeof(float), cudaMemcpyHostToDevice);
    
    *device_x_ptr = device_x;
    *device_y_ptr = device_y;
    *device_k_ptr = device_k;

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
   
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int W_grid = ceil(1.0*W_out / TILE_WIDTH);
    int H_grid = ceil(1.0*H_out / TILE_WIDTH);
    int Z = H_grid * W_grid;

    dim3 dimGrid(B, M, Z);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + K * K);

    conv_forward_kernel<<<dimGrid, dimBlock, shmem_size>>>(device_y, device_x, device_k, B, M, C, H, W, K);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int numOutputElements = B * M * H_out * W_out;
    cudaMemcpy(host_y, device_y, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
