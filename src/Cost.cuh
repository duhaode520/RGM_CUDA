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
    
    void calcuate(Particle* par, double* cost, Flow* data);
    

};

class RegularCost : public Cost {
protected:
    __global__ void execute(double* par, double* cost, Flow* data);
};

class PCost : public Cost {
protected:
    __global__ void execute(double* par, double* cost, Flow* data);
};

class RCost : public Cost {
protected:
    __global__ void execute(double* par, double* cost, Flow* data);
};

#endif