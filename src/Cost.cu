#include "Cost.cuh"
//添加cuda库
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>


Cost::Cost(int nodeNum, int dim, Model* model, MetricsTypeEnum metricsType){
    this->_nodeNum = nodeNum;
    this->_flowNum = nodeNum * (nodeNum - 1);
    this->_dim = dim;
    this->_model = model;
    _metrics = Metrics::create(metricsType);
}

Cost::Cost(int nodeNum, int dim, Model* model, Metrics* metrics) {
    this->_nodeNum = nodeNum;
    this->_flowNum = nodeNum * (nodeNum - 1);
    this->_dim = dim;
    this->_model = model;
    this->_metrics = metrics;
}

Cost::~Cost() {
    delete _metrics;
}


__global__ void kernelWrapper(GlobalConfig* config, double* pars, double* cost, FlowData* data) {
    // 在 host创建的对象的虚函数在device上不能调用，因为虚函数表在host上，所以需要在device上创建对象
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=inherit#data-members
    // 所有的CUDA对象都需要在这个函数中创建

    Model* model = Model::create(config->modelType, config->nodeNum, config->dim);
    Cost* costFunc = Cost::create(config->costType, config->nodeNum, config->dim, model, config->metricsType);
    int flowNum = config->nodeNum * (config->nodeNum - 1);
    for (int i = 0; i < flowNum; i++) {
        int src = data[i].src;
        assert(src < flowNum);
    }

    // for (int i = 0; i < costFunc->_flowNum; i++) {
    //     printf("%d->%d: %lf, %lf\n", data[i].src, data[i].dest, data[i].flow, data[i].dist);
    // }
    
    costFunc->_execute(pars, cost, data); // 这个地方不work，关键原因是Cost的中有很多不在Cuda上的内存，所以不能直接调用

    delete model;
    delete costFunc;
}

void Cost::calculate(GlobalConfig config, double** pars, int parNum, FlowData* data, double* cost) {
    
    // generate linear array
    double* LPar = new double[N_PAR * _dim];
    for (int i = 0; i < N_PAR; i++) {
        for (int j = 0; j < _dim; j++) {
            LPar[i * _dim + j] = pars[i][j];
        }
    }
    // allocate memory on GPU
    double* d_Par = NULL;
    double* d_cost = NULL;
    FlowData* d_data = NULL;
    GlobalConfig* d_config;

    cudaMalloc((void**)&d_cost, N_PAR * sizeof(double));
    cudaMalloc((void**)&d_data, _flowNum * sizeof(FlowData));
    cudaMalloc((void**)&d_Par, N_PAR * _dim * sizeof(double));
    cudaMalloc((void**)&d_config, sizeof(GlobalConfig));

    // copy data from CPU to GPU
    cudaMemcpy(d_data, data, _flowNum * sizeof(FlowData), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Par, LPar, N_PAR * _dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_config, &config, sizeof(GlobalConfig), cudaMemcpyHostToDevice);
    int blockNum = (N_PAR + (_THREADS_PER_BLOCK - 1)) / _THREADS_PER_BLOCK;

    kernelWrapper<<<blockNum, _THREADS_PER_BLOCK>>>
    (d_config, d_Par, d_cost, d_data);
    cudaDeviceSynchronize();

    //* Debug test
    // kernelWrapper<<<1,1>>> (d_config, d_Par, d_cost, d_data);
    cudaMemcpy(cost, d_cost, N_PAR * sizeof(double), cudaMemcpyDeviceToHost);

    // release memory
    cudaFree(d_Par);
    cudaFree(d_cost);
    cudaFree(d_data);
    cudaFree(d_config);


    delete[] LPar;

}

Cost* RegularCost::prepareForDevice() {
    // copy all Cost Members from CPU to GPU 
    RegularCost* d_costFunc;
    cudaMalloc((void**)&d_costFunc, sizeof(RegularCost));
    cudaMemcpy(d_costFunc, this, sizeof(RegularCost), cudaMemcpyHostToDevice);
    Model* d_model = _model->prepareForDevice();
    Metrics* d_metrics = _metrics->prepareForDevice();
    cudaMemcpy(&(d_costFunc->_model), &d_model, sizeof(Model*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_costFunc->_metrics), &d_metrics, sizeof(Metrics*), cudaMemcpyHostToDevice);
    return d_costFunc;
}

Cost* PCost::prepareForDevice() {
    PCost* d_costFunc;
    cudaMalloc((void**)&d_costFunc, sizeof(PCost));

    d_costFunc->_model = _model->prepareForDevice();
    d_costFunc->_metrics = _metrics->prepareForDevice();
    return d_costFunc;
}

void Cost::leaveDevice() {
    _metrics->leaveDevice();
    cudaFree(_metrics);
    _model->leaveDevice();
    cudaFree(_model);
}

void Cost::predict(double* pars, FlowData* data, int metricsSize, MetricsTypeEnum metricsTypes[], double* cost) {
    double* pred = new double[dataConfig->flowNum];
    _model->pred(0, pars, pred, data);
    Metrics* m;
    for (int i = 0; i < metricsSize; i++) {
        m = Metrics::create(metricsTypes[i]);
        cost[i] = m->calc(data, pred, dataConfig->flowNum);
        delete m;
    }
    delete pred;
}

// 对应的是 gh 代码的 cost
__device__ void RegularCost::_execute(double* pars, double *cost, FlowData* data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double* pred;
    int flowNum = _nodeNum * (_nodeNum - 1);
    cudaMalloc((void**)&pred, flowNum * sizeof(double));

    for (int i = 0; i < flowNum; i++) {
        int src = data[i].src;
        if (src > flowNum) {
            printf("execute: flow %d is changed in kernel %d\n", i, index);
        }
    }

    _model->pred(index, pars, pred, data);
    cost[index] = _metrics->calc(data, pred, flowNum);
    cudaFree(pred);
}

__device__ __host__ Cost* Cost::create(CostTypeEnum costType, int nodeNum, int dim, Model* model, MetricsTypeEnum metricsType) {
    switch (costType) {
    case CostTypeEnum::Regular:
        return new RegularCost(nodeNum, dim, model, metricsType);
    case CostTypeEnum::P:
        return new PCost(nodeNum, dim, model, metricsType);
    default:
        printf("Error: Cost Type Error, return null by default\n");
        return nullptr;
    }
}

// void Cost::destroy(Cost* cost) {
//     cost->~Cost();
//     cudaFree(cost);
// }

__device__ void PCost::_execute(double* pars, double* cost, FlowData* data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double* pred;
    int flowNum = _nodeNum * (_nodeNum - 1);
    cudaMalloc((void**)&pred, flowNum * sizeof(double));
    _model->pred(index, pars, pred, data);
    cost[index] = _metrics->calc(data, pred, flowNum);
    cudaFree(pred);
}
