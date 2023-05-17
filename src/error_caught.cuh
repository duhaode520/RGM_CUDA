#ifndef ERROR_CUH
#define ERROR_CUH

#include "Flow.h"
#include "PSOConfig.h"
__device__ __host__ 
void checkCudaErrors(cudaError_t err, int index, FlowData* data, const char* file, const int line);

#endif
