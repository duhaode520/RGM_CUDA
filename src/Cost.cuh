#ifndef COST_H
#define COST_H
#include "Flow.h"
#include "Metrics.cuh"
#include "PSOConfig.h"
#include "Model.cuh"
#include "consts.h"
#include <cuda_runtime.h>



/**
 * @brief parent class for all cost functions
 */
class Cost {
protected:
    /* data */
    Metrics* _metrics;
    Model* _model;
    int _nodeNum;
    int _dim;
    int _flowNum;

    __device__ virtual void _execute(float* pars, float* cost, FlowData* data) = 0;


    static const int _THREADS_PER_BLOCK = 32; // thread number per block used in kernel function

public:
    __device__ __host__ Cost() {}
    __device__ __host__ Cost(int nodeNum, int dim, int flowNum, Model* model, MetricsTypeEnum metricsType); 
    __device__ __host__ Cost(int nodeNum, int dim, int flowNum, Model* model, Metrics* metrics);
    __device__ __host__ virtual ~Cost(); 
    virtual Cost* prepareForDevice() = 0;

    void leaveDevice();
    
    void calculate(GlobalConfig cfg, float** pars, int parNum, FlowData* data, float* cost);
    
    /**
     * @brief predict the cost of a particle
     * 
     * @param pars particles
     * @param data  flow data
     * @param metricsSize  the number of metrics
     * @param metricsTypes the types of metrics
     * @param cost the costs of different metrics of the particle
     */
    void predict(float* pars, FlowData* data, int metricsSize, MetricsTypeEnum metricsTypes[], float* cost);

    static __device__ __host__ Cost* create(CostTypeEnum costType, int nodeNum, int dim, int flowNum, Model* model, MetricsTypeEnum metricsType);
    // static __device__ __host__ Cost* create(CostConfig* config, Model* model);
    // static void destroy(Cost* cost);
    
    friend __global__ void kernelWrapper(GlobalConfig* config, float* pars, float* cost, FlowData* data);
};
__global__ void kernelWrapper(GlobalConfig* config, float* pars, float* cost, FlowData* data);

class RegularCost : public Cost {
protected:
    __device__ void _execute(float* par, float* cost, FlowData* data) override;

public:
    __device__ __host__ RegularCost(int nodeNum, int dim, int flowNum, Model* model, MetricsTypeEnum metricsType) 
        : Cost(nodeNum, dim, flowNum, model, metricsType) {}; 
    __device__ __host__ RegularCost(int nodeNum, int dim, int flowNum, Model* model, Metrics* metrics) 
        : Cost(nodeNum, dim, flowNum, model, metrics) {};
    Cost* prepareForDevice();
};

class PCost : public Cost {
protected:
    __device__ void _execute(float* par, float* cost, FlowData* data) override;

public:
    __device__ __host__ PCost(int nodeNum, int dim, int flowNum, Model* model, MetricsTypeEnum metricsType) 
        : Cost(nodeNum, dim, flowNum, model, metricsType) {}; 
    __device__ __host__ PCost(int nodeNum, int dim, int flowNum, Model* model, Metrics* metrics) 
        : Cost(nodeNum, dim, flowNum, model, metrics) {};
    Cost* prepareForDevice();
};


#endif