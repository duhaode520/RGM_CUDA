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


__global__ void kernelWrapper(Cost* costFunc, double* pars, double* cost, Flow* data) {
    costFunc->execute(pars, cost, data);
}

void Cost::calculate(double** pars, int parNum, Flow* data, double* cost) {
    
    // generate linear array
    double* LPar = new double[N_PAR * dim];
    for (int i = 0; i < N_PAR; i++) {
        for (int j = 0; j < dim; j++) {
            LPar[i * dim + j] = pars[i][j];
        }
    }

    // allocate memory on GPU
    double* d_Par;
    double* d_cost;
    cudaMalloc((void**)&d_Par, N_PAR * dim * sizeof(double));
    cudaMalloc((void**)&d_cost, N_PAR * sizeof(double));
    
    // copy data from CPU to GPU
    cudaMemcpy(d_Par, LPar, N_PAR * dim * sizeof(double), cudaMemcpyHostToDevice);

    kernelWrapper<<<(N_PAR + (THREADS_PER_BLOCK + 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
    (this, d_Par, cost, data);
    cudaMemcpy(cost, d_cost, N_PAR * sizeof(double), cudaMemcpyDeviceToHost);

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
__device__ void RegularCost::execute(double* pars, double *cost, Flow* data) {
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

__device__ void PCost::execute(double* pars, double* cost, Flow* data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double* pred;
    int flowNum = nodeNum * (nodeNum - 1) / 2;
    cudaMalloc((void**)&pred, flowNum * sizeof(double));
    model->pred(index, pars, pred, data);
    cost[index] = metrics->calc(data, pred, flowNum);
    cudaFree(pred);
}
