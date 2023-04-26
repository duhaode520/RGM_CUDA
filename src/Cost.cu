#include "Cost.cuh"
//添加cuda库
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void Cost::execute(Particle* par, double *cost, Flow* data) {
}

void Cost::calcuate(Particle* par, double* cost, Flow* data) {
    
}
