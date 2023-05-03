#ifndef COST_H
#define COST_H
#include "Flow.h"
#include "Metrics.cuh"
#include "PSOConfig.h"
#include "Model.cuh"
#include "consts.h"
#include "cuda_runtime.h"



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
    Cost() {}
    Cost(int nodeNum, int dim, Model* model, MetricsTypeEnum metricsType); 
    ~Cost(); 
    
    /**
     * @brief Calculate the cost of particles
     * 
     * @param pars all particles parameters
     * @param parNum the number of particles
     * @param cost return the cost of each particle 
     * @param data flow datas
     */
    void calculate(double** pars, int parNum, Flow* data, double* cost);
    
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
    
    friend __global__ void kernelWrapper(Cost* costFunc, double* pars, double* cost, Flow* data);
};
__global__ void kernelWrapper(Cost* costFunc, double* pars, double* cost, Flow* data);

class RegularCost : public Cost {
protected:
    __device__ void execute(double* par, double* cost, Flow* data) override;

public:
    RegularCost(int nodeNum, int dim, Model* model, MetricsTypeEnum metricsType); 
};

class PCost : public Cost {
protected:
    __device__ void execute(double* par, double* cost, Flow* data) override;

public:
    PCost(int nodeNum, int dim, Model* model, MetricsTypeEnum metricsType); 
};


#endif