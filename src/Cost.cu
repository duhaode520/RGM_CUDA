#include "Cost.cuh"
//添加cuda库
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


Cost::Cost(int nodeNum, int dim, Model* model, MetricsTypeEnum metricsType){
    this->nodeNum = nodeNum;
    this->dim = dim;
    this->model = model;
    Metrics::create(metrics, metricsType);
}

Cost::~Cost() {
    Metrics::destroy(metrics);
}


__global__ void kernelWrapper(Cost* costFunc, double* pars, double* cost, Flow* data) {
    //TODO: prepare data for device, see 
    // https://stackoverflow.com/questions/39006348/accessing-class-data-members-from-within-cuda-kernel-how-to-design-proper-host
    // https://stackoverflow.com/questions/65325842/how-do-i-properly-implement-classes-whose-members-are-called-both-from-host-and?noredirect=1&lq=1
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
    double* d_Par = NULL;
    double* d_cost = NULL;
    Flow* d_data = NULL;
    cudaMalloc((void**)&d_cost, N_PAR * sizeof(double));
    cudaMalloc((void**)&d_data, dataConfig->flowNum * sizeof(Flow));
    // FIXME: this malloc turns d_Par to NULL
    cudaMalloc((void**)&d_Par, N_PAR * dim * sizeof(double));
    if (d_Par == NULL) {
        throw std::runtime_error("Failed to allocate memory on GPU");
    }
    // copy data from CPU to GPU
    cudaMemcpy(d_data, data, dataConfig->flowNum * sizeof(Flow), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Par, LPar, N_PAR * dim * sizeof(double), cudaMemcpyHostToDevice);

    kernelWrapper<<<(N_PAR + (THREADS_PER_BLOCK + 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
    (this, d_Par, d_cost, d_data);
    cudaDeviceSynchronize();
    cudaMemcpy(cost, d_cost, N_PAR * sizeof(double), cudaMemcpyDeviceToHost);

    // release memory
    cudaFree(d_Par);
    cudaFree(d_cost);
    cudaFree(d_data);
    delete[] LPar;

}

void Cost::predict(double* pars, Flow* data, int metricsSize, MetricsTypeEnum metricsTypes[], double* cost) {
    double* pred = new double[dataConfig->flowNum];
    int flowNum = nodeNum * (nodeNum - 1) / 2;
    model->pred(0, pars, pred, data);
    Metrics* m;
    for (int i = 0; i < metricsSize; i++) {
        Metrics::create(m, metricsTypes[i]);
        cost[i] = m->calc(data, pred, flowNum);
        Metrics::destroy(m);
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

void Cost::create(Cost* cost, CostTypeEnum costType, int nodeNum, int dim, Model* model, MetricsTypeEnum metricsType) {
    switch (costType) {
    case CostTypeEnum::Regular:
        cudaMallocManaged((void**)&cost, sizeof(RegularCost));
        new(cost) RegularCost(nodeNum, dim, model, metricsType);
        break;
    case CostTypeEnum::P:
        cudaMallocManaged((void**)&cost, sizeof(PCost));
        new(cost) PCost(nodeNum, dim, model, metricsType);
        break;
    default:
        throw std::runtime_error("Unknown cost type");
    }
}

void Cost::destroy(Cost* cost) {
    cost->~Cost();
    cudaFree(cost);
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
