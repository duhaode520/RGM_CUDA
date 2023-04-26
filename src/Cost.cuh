#ifndef COST_H
#define COST_H
#include "Flow.h"
#include "Particle.h"
#include "Metrics.h"

/**
 * @brief parent class for all cost functions
 * 
 */
class Cost {
protected:
    /* data */
    Metrics metrics;
    // TODO: 这里可能还要统一做一些修改,核函数的参数应该都是简单变量吧
    virtual __global__ void execute(Particle* par, double* cost, Flow* data);

public:
    Cost(/* args */) {}
    ~Cost() {}
    
    void calcuate(Particle* par, double* cost, Flow* data);
    

};

class RegularCost : public Cost {
protected:
    __global__ void execute(Particle* par, double* cost, Flow* data);
};

class PCost : public Cost {
protected:
    __global__ void execute(Particle* par, double* cost, Flow* data);
};

class RCost : public Cost {
protected:
    __global__ void execute(Particle* par, double* cost, Flow* data);
};

#endif