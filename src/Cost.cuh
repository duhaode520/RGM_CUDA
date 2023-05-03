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
 * @todo Model 和 Cost 现在是完全一一对应的，未来希望能把Model独立出来处理
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
    Cost(int nodeNum, int dim, Model* model, MetricsTypeEnum metricsType); 
    ~Cost() {}
    
    /**
     * @brief Calculate the cost of particles
     * 
     * @param par all particles
     * @param cost return the cost of each particle 
     * @param data flow datas
     */
    void calculate(Particle* par, double* cost, Flow* data); // TODO: 从 Particle 解耦
    
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
    

};

class RegularCost : public Cost {
protected:
    __global__ void execute(double* par, double* cost, Flow* data);

public:
    RegularCost(int nodeNum, int dim, Model* model, MetricsTypeEnum metricsType); 
};

class PCost : public Cost {
protected:
    __global__ void execute(double* par, double* cost, Flow* data);

public:
    PCost(int nodeNum, int dim, Model* model, MetricsTypeEnum metricsType); 
};


#endif