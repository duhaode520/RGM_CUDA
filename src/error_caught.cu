#include "cuda_runtime.h"
#include "error_caught.cuh"
__device__ __host__ void checkCudaErrors(cudaError_t err, int index, FlowData* data, const char* file, const int line) {
    if (err != cudaSuccess) {
        printf("CUDA Thread %d / %d : error at %s:%d code=%s (%s) \n", index, N_PAR,  file, line, cudaGetErrorName(err), cudaGetErrorString(err));
        // cudaDeviceReset();
    }
}

