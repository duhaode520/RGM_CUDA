#ifndef PARTICLE_H
#define PARTICLE_H

#include "utils.h"
#include "Flow.h"
#include "Cost.cuh"
#include "consts.h"
#include "PSOConfig.h"

class Particle {
private:
    /* data */
    // int Npar; // number of particles
    int dim; // dimension of particles
    double** Par;
    double** Pbest;
    double Pbest_cost[N_PAR];
    double* Lpar;
    double cost[N_PAR];

    double* Gbest;
    double Gbest_cost;
    Cost* costFunction;
    Model* model;
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

    void setCost(CostTypeEnum costType, MetricsTypeEnum metricsType);
    
    void setModel(ModelTypeEnum modelType);
    // 训练
    void train(Flow* data);

    // 获取最后Gbest的cost
    void predictCost(Flow* data, double* cost);
    
    // 获取全局最优解对应的cost
    double getGbestCost();

    // 获取全局最优解的beta
    double getGbestBeta();

    std::string getResult();
    
    // static constexpr double Xmin = 0.001;
    // static constexpr double Xmax = 100000000;
    // static constexpr double Xrandmin = 5;
    // static constexpr double Xrandmax = 70;
    // static constexpr int Npar = 4096;// Number of particles each swarm
    // static constexpr double alpha = 0.75;
    // static constexpr double pjump = 0.001;
    // static constexpr double SCALE = 1;
    // static constexpr int Maxiter = 3000;
    static constexpr int MetricsNum = 2;

    friend class Cost;
};



#endif