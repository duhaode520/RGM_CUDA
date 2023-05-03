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
    costFunc->execute(pars, cost, data); // 这个地方不work，关键原因是Cost的中有很多不在Cuda上的内存，所以不能直接调用
    
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
    Flow* d_data;
    Cost* d_costFunc;
    cudaMalloc((void**)&d_Par, N_PAR * dim * sizeof(double));
    cudaMalloc((void**)&d_cost, N_PAR * sizeof(double));
    cudaMalloc((void**)&d_data, dataConfig->flowNum * sizeof(Flow));
    cudaMalloc((void**)&d_costFunc, sizeof(Cost));
    
    // copy data from CPU to GPU
    cudaMemcpy(d_Par, LPar, N_PAR * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, data, dataConfig->flowNum * sizeof(Flow), cudaMemcpyHostToDevice);
    cudaMemcpy(d_costFunc, this, sizeof(Cost), cudaMemcpyHostToDevice);

    kernelWrapper<<<(N_PAR + (THREADS_PER_BLOCK + 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
    (d_costFunc, d_Par, d_cost, d_data);
    cudaDeviceSynchronize();
    cudaMemcpy(cost, d_cost, N_PAR * sizeof(double), cudaMemcpyDeviceToHost);

    // release memory
    cudaFree(d_Par);
    cudaFree(d_cost);
    cudaFree(d_data);
    cudaFree(d_costFunc);
    delete[] LPar;

}

void Cost::predict(double* pars, Flow* data, int metricsSize, MetricsTypeEnum metricsTypes[], double* cost) {
    double* pred = new double[dataConfig->flowNum];
    int flowNum = nodeNum * (nodeNum - 1) / 2;
    model->pred(0, pars, pred, data);
    for (int i = 0; i < metricsSize; i++) {
        metrics = Metrics::createMetrics(metricsTypes[i]);
        cost[i] = metrics->calc(data, pred, flowNum);
    }
    delete pred;
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
