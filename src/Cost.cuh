#ifndef COST_H
#define COST_H
#include "Flow.h"
#include "Metrics.cuh"
#include "PSOConfig.h"
#include "Model.cuh"
#include "consts.h"
#include "cuda_runtime.h"


struct CostConfig {
    int nodeNum;
    int dim;
    CostTypeEnum costType;
    ModelTypeEnum modelType;
    MetricsTypeEnum metricsType;
};

/**
 * @brief parent class for all cost functions
 */
class Cost {
protected:
    /* data */
    Metrics* metrics;
    Model* model;
    int nodeNum;
    int dim;

    __device__ virtual void execute(double* pars, double* cost, Flow* data) = 0;


    static const int THREADS_PER_BLOCK = 64; // thread number per block used in kernel function

public:
    __device__ __host__ Cost() {}
    __device__ __host__ Cost(int nodeNum, int dim, Model* model, MetricsTypeEnum metricsType); 
    __device__ __host__ Cost(int nodeNum, int dim, Model* model, Metrics* metrics);
    __device__ __host__ virtual ~Cost(); 
    virtual Cost* prepareForDevice() = 0;

    void leaveDevice();
    
    void calculate(CostConfig cfg, double** pars, int parNum, Flow* data, double* cost);
    
    /**
     * @brief predict the cost of a particle
     * 
     * @param pars particles
     * @param data  flow data
     * @param metricsSize  the number of metrics
     * @param metricsTypes the types of metrics
     * @param cost the costs of different metrics of the particle
     */
    void predict(double* pars, Flow* data, int metricsSize, MetricsTypeEnum metricsTypes[], double* cost);

    static __device__ __host__ Cost* create(CostTypeEnum costType, int nodeNum, int dim, Model* model, MetricsTypeEnum metricsType);

    // static void destroy(Cost* cost);
    
    friend __global__ void kernelWrapper(CostConfig* config, double* pars, double* cost, Flow* data);
};
__global__ void kernelWrapper(CostConfig* config, double* pars, double* cost, Flow* data);

class RegularCost : public Cost {
protected:
    __device__ void execute(double* par, double* cost, Flow* data) override;

public:
    __device__ __host__ RegularCost(int nodeNum, int dim, Model* model, MetricsTypeEnum metricsType) 
        : Cost(nodeNum, dim, model, metricsType) {}; 
    __device__ __host__ RegularCost(int nodeNum, int dim, Model* model, Metrics* metrics) 
        : Cost(nodeNum, dim, model, metrics) {};
    Cost* prepareForDevice();
};

class PCost : public Cost {
protected:
    __device__ void execute(double* par, double* cost, Flow* data) override;

public:
    __device__ __host__ PCost(int nodeNum, int dim, Model* model, MetricsTypeEnum metricsType) 
        : Cost(nodeNum, dim, model, metricsType) {}; 
    __device__ __host__ PCost(int nodeNum, int dim, Model* model, Metrics* metrics) 
        : Cost(nodeNum, dim, model, metrics) {};
    Cost* prepareForDevice();
};


#endif