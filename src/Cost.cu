#include "Cost.cuh"
//添加cuda库
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


Cost::Cost(int nodeNum, int dim, Model* model, MetricsTypeEnum metricsType){
    this->nodeNum = nodeNum;
    this->dim = dim;
    this->model = model;
    this->metrics = Metrics::createMetrics(metricsType);
}

Cost::~Cost() {
    delete metrics;
}

void Cost::calculate(Particle* particles, double* cost, Flow* data) {
    
    // generate linear array
    double* LPar = new double[particles->Npar * particles->dim];
    for (int i = 0; i < particles->Npar; i++) {
        for (int j = 0; j < particles->dim; j++) {
            LPar[i * particles->dim + j] = particles->Par[i][j];
        }
    }

    // allocate memory on GPU
    double* d_Par;
    double* d_cost;
    cudaMalloc((void**)&d_Par, particles->Npar * dim * sizeof(double));
    cudaMalloc((void**)&d_cost, particles->Npar * sizeof(double));
    
    // copy data from CPU to GPU
    cudaMemcpy(d_Par, LPar, particles->Npar * dim * sizeof(double), cudaMemcpyHostToDevice);

    execute<<<(particles->Npar + (THREADS_PER_BLOCK + 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
    (d_Par, cost, data);
    cudaMemcpy(cost, d_cost, particles->Npar * sizeof(double), cudaMemcpyDeviceToHost);

    // release memory
    cudaFree(d_Par);
    cudaFree(d_cost);
    delete[] LPar;

}

void Cost::predict(double* pars, Flow* data, int metricsSize, MetricsTypeEnum metricsTypes[], double* cost) {
    double* pred;
    int flowNum = nodeNum * (nodeNum - 1) / 2;
    cudaMalloc((void**)&pred, flowNum * sizeof(double));
    model->pred(0, pars, pred, data);
    for (int i = 0; i < metricsSize; i++) {
        metrics = Metrics::createMetrics(metricsTypes[i]);
        cost[i] = metrics->calc(data, pred, flowNum);
    }
    cudaFree(pred);
}

// 对应的是 gh 代码的 cost
__global__ void RegularCost::execute(double* pars, double *cost, Flow* data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double* pred;
    int flowNum = nodeNum * (nodeNum - 1) / 2;
    cudaMalloc((void**)&pred, flowNum * sizeof(double));
    model->pred(index, pars, pred, data);
    cost[index] = metrics->calc(data, pred, flowNum);
    cudaFree(pred);
}

RegularCost::RegularCost(int nodeNum, int dim, Model* model, MetricsTypeEnum metricsType) 
    : Cost(nodeNum, dim, model, metricsType) {
}

PCost::PCost(int nodeNum, int dim, Model* model, MetricsTypeEnum metricsType) 
    : Cost(nodeNum, dim, model, metricsType) {
}

__global__ void PCost::execute(double* pars, double* cost, Flow* data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double* pred;
    int flowNum = nodeNum * (nodeNum - 1) / 2;
    cudaMalloc((void**)&pred, flowNum * sizeof(double));
    model->pred(index, pars, pred, data);
    cost[index] = metrics->calc(data, pred, flowNum);
    cudaFree(pred);
}
