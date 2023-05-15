#ifndef PARTICLE_H
#define PARTICLE_H

#include "utils.h"
#include "Flow.h"
#include "Cost.cuh"
#include "consts.h"
#include "PSOConfig.h"


class Particle {
protected:
    /* data */
    // int Npar; // number of particles
    int _par_dim; // dimension of particles
    float** _Par;
    float** _Pbest;
    float _Pbest_cost[N_PAR];
    float* _Lpar;
    float _cost[N_PAR];

    Cost* _cost_func;
    Model* _model;
    GlobalConfig _config;

    float* _Gbest; // global best for particles
    float _Gbest_cost;
    int _Gbest_id;

    virtual void _particle_init(int dim);

    void _setCost(CostTypeEnum costType, MetricsTypeEnum metricsType);
    
    void _setModel(ModelTypeEnum modelType);

    // Cost and train
    bool _cost_init = false;

    void _costInitialize(FlowData* data);

    virtual void _costCalc(FlowData* data);

    virtual void _bestUpdate();

    void _swarmUpdate();

public:

    Particle(){};
    virtual ~Particle();

    /**
     * @brief Initialize the particle
     * 
     * @param dim dimension of particles
     * @param costFunction pointer of the cost function
     * @param data pointer of the flow data
     * @attention This function must be called before train() function,
     * and must be called for each particle
     */

    virtual void initialize(GlobalConfig config);

    // 训练
    virtual void train(FlowData* data);

    // 获取最后Gbest的cost
    void predictCost(FlowData* data, float* cost);
    
    // 获取全局最优解对应的cost
    float getGbestCost();

    // 获取全局最优解的beta
    float getGbestBeta();

    std::string getResult();

    /**
     * @brief Cooperate with other particle, used in CPSO-H
     * 从另一个粒子里的Gbest拿来作为自己的一个粒子
     * 
     * @param other 
     */
    void cooperate(Particle* other);
    
    static constexpr int MetricsNum = 2;

    friend class Cost;
};

class PParticle : public Particle {
private:
    void __copyToPGbest();
protected:

    void _costCalc(FlowData* data) override;

    void _bestUpdate() override;

public:
    PParticle() {};
    
    void initialize(GlobalConfig config) override;

    static float* PGbest; // 全部 P 类型粒子在全部维度上的最优解

};

#endif