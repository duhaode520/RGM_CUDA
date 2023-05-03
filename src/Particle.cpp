#include "Particle.h"
#include <cstring>

void Particle::initialize(int dim) {
    this->dim = dim;

    // 1. particles initialization
    Par = new double*[N_PAR];
    Pbest = new double*[N_PAR];
    for (int i = 0; i < N_PAR; i++) {
        Par[i] = new double[dim];
        Pbest[i] = new double[dim];
    }

    for (int i = 0; i < N_PAR; i++) {
        for (int d = 0; d < dim; d++) {
            if (Flow::tflow[d] > 0) {
                Par[i][d] = X_RAND_MIN + (X_RAND_MAX - X_RAND_MIN) * rand() / RAND_MAX;
            } else {
                Par[i][d] = 0;
            }
            Pbest[i][d] = Par[i][d];
        }
    }
    Gbest = new double[dim];
    for (int d = 0; d < dim; d++) {
        Gbest[d] = Pbest[0][d];
    }

}

void Particle::setCost(CostTypeEnum costType, MetricsTypeEnum metricsType) {
    switch (costType) {
    case CostTypeEnum::Regular:
        this->costFunction = new RegularCost(dataConfig.nodeNum, dim, model, metricsType);  
        break;
    case CostTypeEnum::P:
        this->costFunction = new PCost(dataConfig.nodeNum, dim, model, metricsType);  
        break;
    default:
        throw "Unknown Cost Type";
        break;
    }
}

void Particle::setModel(ModelTypeEnum modelType) {
    this->model = Model::createModel(modelType, dataConfig.nodeNum, dim);
}

void Particle::train(Flow* data) {
    if (!cost_init) {
        cost_init = true;
        costInitialize(data);
        return;
    }
    costFunction->calculate(Par, N_PAR, data, cost);
    bestUpdate();
    swarmUpdate();
}

void Particle::predictCost(Flow *data, double *cost) {
    MetricsTypeEnum metricsTypes[MetricsNum] 
        = {MetricsTypeEnum::RMSE, MetricsTypeEnum::R2};
    costFunction->predict(Gbest, data, MetricsNum, metricsTypes, cost);

}

void Particle::costInitialize(Flow *data) {
    // cost initialization
    costFunction->calculate(Par, N_PAR, data, cost);
    memcpy(Pbest_cost, cost, sizeof(double) * N_PAR);
    Gbest_cost = Pbest_cost[0];
    Gbest_id = 0;
}

void Particle::bestUpdate() {
    for (int p = 0; p < N_PAR; p++) {
        int cur_cost = cost[p];
        if (cur_cost < Pbest_cost[p]) {
            for (int d = 0; d < dim; d++) {
                Pbest[p][d] = Par[p][d];
            }
            Pbest_cost[p] = cur_cost;
        }

        if (cur_cost < Gbest_cost) {
            for (int d = 0; d < dim; d++) {
                Gbest[d] = Par[p][d];
            }
            Gbest_cost = cur_cost;
            Gbest_id = p;
        }
    }
}

void Particle::swarmUpdate() {
    // TODO: 未来如果要引入不同的 update 方式可以把这个函数改成类
    // TODO: 这一部分可以用CUDA加速
    for (int p = 0; p < N_PAR; p++) {
        for (int d = 0; d < dim; d++) {
            if (Flow::tflow[d] == 0) {
                continue;
            }
            double sigma = abs(Gbest[d] - Par[p][d]);
            Par[p][d] = Gbest[d] + ALPHA * sigma * BoxMullerRandom();
            double rjump = 1.0 * random01();
            if(rjump<P_JUMP) {
                Par[p][d]=1.0 * random01() *
                 (X_RAND_MAX - X_RAND_MIN) + X_RAND_MIN;
            }

            if(Par[p][d] > X_MAX) {
                Par[p][d] = X_MAX;
            }
            else if(Par[p][d] < X_MIN) {
                Par[p][d] = X_MIN;
            }
        }
    }
}

double Particle::getGbestCost() {
    return Gbest_cost;
}

double Particle::getGbestBeta() {
    return Gbest[dim - 1];
}

std::string Particle::getResult() { 
    return model->getResult(Gbest);
}

Particle::~Particle() {
    delete [] Gbest;
    for (int i = 0; i < N_PAR; i++) {
        delete [] Par[i];
        delete [] Pbest[i];
    }
    delete [] Par;
    delete [] Pbest;

    delete costFunction;
}
