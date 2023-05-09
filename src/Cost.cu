#include "Cost.cuh"
//添加cuda库
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


Cost::Cost(int nodeNum, int dim, Model* model, MetricsTypeEnum metricsType){
    this->nodeNum = nodeNum;
    this->dim = dim;
    this->model = model;
    metrics = Metrics::create(metricsType);
}

Cost::Cost(int nodeNum, int dim, Model* model, Metrics* metrics) {
    this->nodeNum = nodeNum;
    this->dim = dim;
    this->model = model;
    this->metrics = metrics;
}

Cost::~Cost() {
    delete metrics;
}


__global__ void kernelWrapper(CostConfig* config, double* pars, double* cost, Flow* data) {
    // FIXME: 在 host创建的对象的虚函数在device上不能调用，因为虚函数表在host上，所以需要在device上创建对象
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=inherit#data-members
    // TODO: 所有的CUDA对象都需要在这个函数中创建
    Model* model = Model::create(config->modelType, config->nodeNum, config->dim);
    Cost* costFunc = Cost::create(config->costType, config->nodeNum, config->dim, model, config->metricsType);

    costFunc->execute(pars, cost, data); // 这个地方不work，关键原因是Cost的中有很多不在Cuda上的内存，所以不能直接调用

    delete model;
    delete costFunc;
}

void Cost::calculate(CostConfig config, double** pars, int parNum, Flow* data, double* cost) {
    
    // generate linear array
    double* LPar = new double[N_PAR * dim];
    for (int i = 0; i < N_PAR; i++) {
        for (int j = 0; j < dim; j++) {
            LPar[i * dim + j] = pars[i][j];
        }
    }
    std::cout << typeid(*this).name() << std::endl;
    // allocate memory on GPU
    double* d_Par = NULL;
    double* d_cost = NULL;
    Flow* d_data = NULL;
    CostConfig* d_config;

    cudaMalloc((void**)&d_cost, N_PAR * sizeof(double));
    cudaMalloc((void**)&d_data, dataConfig->flowNum * sizeof(Flow));
    cudaMalloc((void**)&d_Par, N_PAR * dim * sizeof(double));
    cudaMalloc((void**)&d_config, sizeof(CostConfig));

    // copy data from CPU to GPU
    cudaMemcpy(d_data, data, dataConfig->flowNum * sizeof(Flow), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Par, LPar, N_PAR * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_config, &config, sizeof(CostConfig), cudaMemcpyHostToDevice);

    kernelWrapper<<<(N_PAR + (THREADS_PER_BLOCK + 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
    (d_config, d_Par, d_cost, d_data);
    cudaDeviceSynchronize();
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
    Model* d_model = model->prepareForDevice();
    Metrics* d_metrics = metrics->prepareForDevice();
    cudaMemcpy(&(d_costFunc->model), &d_model, sizeof(Model*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_costFunc->metrics), &d_metrics, sizeof(Metrics*), cudaMemcpyHostToDevice);
    return d_costFunc;
}

Cost* PCost::prepareForDevice() {
    PCost* d_costFunc;
    cudaMalloc((void**)&d_costFunc, sizeof(PCost));

    d_costFunc->model = model->prepareForDevice();
    d_costFunc->metrics = metrics->prepareForDevice();
    return d_costFunc;
}

void Cost::leaveDevice() {
    metrics->leaveDevice();
    cudaFree(metrics);
    model->leaveDevice();
    cudaFree(model);
}

void Cost::predict(double* pars, Flow* data, int metricsSize, MetricsTypeEnum metricsTypes[], double* cost) {
    double* pred = new double[dataConfig->flowNum];
    int flowNum = nodeNum * (nodeNum - 1) / 2;
    model->pred(0, pars, pred, data);
    Metrics* m;
    for (int i = 0; i < metricsSize; i++) {
        m = Metrics::create(metricsTypes[i]);
        cost[i] = m->calc(data, pred, flowNum);
        delete m;
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

__device__ void PCost::execute(double* pars, double* cost, Flow* data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double* pred;
    int flowNum = nodeNum * (nodeNum - 1) / 2;
    cudaMalloc((void**)&pred, flowNum * sizeof(double));
    model->pred(index, pars, pred, data);
    cost[index] = metrics->calc(data, pred, flowNum);
    cudaFree(pred);
}
