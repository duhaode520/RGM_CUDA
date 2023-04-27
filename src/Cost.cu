#include "Cost.cuh"
//添加cuda库
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// 对应的是 gh 代码的 cost
__global__ void Cost::execute(Particle* par, double *cost, Flow* data) {
    
}

void Cost::calcuate(Particle* par, double* cost, Flow* data) {
    // 空间的分配
    execute<<<1, 1>>>(par, cost, data);
    // 空间的释放
}
