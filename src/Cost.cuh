#ifndef COST_H
#define COST_H
#include "Flow.h"
#include "Particle.h"
#include "Metrics.cuh"
#include "PSOConfig.h"
#include "Model.cuh"
#include "consts.h"

/**
 * @brief parent class for all cost functions
 * 
 */
class Cost {
protected:
    /* data */
    Metrics* metrics;
    Model* model;
    int nodeNum;
    int dim;

    __global__ virtual void execute(double* pars, double* cost, Flow* data);

    static const int THREADS_PER_BLOCK = 64; // thread number per block used in kernel function

public:
    Cost() {}
    Cost(int nodeNum, int dim, ModelTypeEnum modelType, MetricsTypeEnum metricsType); 
    ~Cost() {}
    
    /**
     * @brief Calculate the cost of a particle
     * 
     * @param par all particles
     * @param cost return the cost of each particle 
     * @param data flow datas
     */
    void calcuate(Particle* par, double* cost, Flow* data);
    

};

class RegularCost : public Cost {
protected:
    __global__ void execute(double* par, double* cost, Flow* data);

public:
    RegularCost(int nodeNum, int dim, ModelTypeEnum modelType, MetricsTypeEnum metricsType); 
};

class PCost : public Cost {
protected:
    __global__ void execute(double* par, double* cost, Flow* data);

public:
    PCost(int nodeNum, int dim, ModelTypeEnum modelType, MetricsTypeEnum metricsType); 
};

class RCost : public Cost {
protected:
    __global__ void execute(double* par, double* cost, Flow* data);
};

#endif