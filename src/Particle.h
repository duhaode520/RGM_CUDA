#ifndef PARTICLE_H
#define PARTICLE_H

#include "utils.h"
#include "Flow.h"
#include "Cost.cuh"
#include "consts.h"

class Particle {
private:
    /* data */
    // int Npar; // number of particles
    int dim; // dimension of particles
    double** Par;
    double** Pbest;
    double Pbest_cost[Npar];
    double* Lpar;
    double cost[Npar];

    double* Gbest;
    double Gbest_cost;
    Cost costFunction;
    int Gbest_id;

    bool cost_init = false;

    void costInitialize(Flow* data);

    void bestUpdate();
    void swarmUpdate();

public:

    Particle(){};
    ~Particle();

    /**
     * @brief Initialize the particle
     * 
     * @param dim dimension of particles
     * @param costFunction pointer of the cost function
     * @param data pointer of the flow data
     * @attention This function must be called before train() function,
     * and must be called for each particle
     */
    void initialize(int dim);

    void setCost(CostTypeEnum costType, ModelTypeEnum modelType, MetricsTypeEnum metricsType);
    
    // 训练
    void train(Flow* data);

    // 获取全局最优解对应的cost
    double getGbestCost();

    // 获取全局最优解的beta
    double getGbestBeta();
    
    static const double Xmin = 0.001;
    static const double Xmax = 100000000;
    static const double Xrandmin = 5;
    static const double Xrandmax = 70;
    static const int Npar = 4096;// Number of particles each swarm
    static const double alpha = 0.75;
    static const double pjump = 0.001;
    static const double SCALE = 1;
    static const int Maxiter = 3000;

    friend class Cost;
};



#endif