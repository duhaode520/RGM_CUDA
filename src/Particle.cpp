#include "Particle.h"
#include <cstring>

void Particle::initialize(int dim) {
    this->dim = dim;

    // 1. particles initialization
    Par = new double*[Npar];
    Pbest = new double*[Npar];
    for (int i = 0; i < Npar; i++) {
        Par[i] = new double[dim];
        Pbest[i] = new double[dim];
    }

    for (int i = 0; i < Npar; i++) {
        for (int d = 0; d < dim; d++) {
            if (Flow::tflow[d] > 0) {
                Par[i][d] = Xrandmin + (Xrandmax - Xrandmin) * rand() / RAND_MAX;
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
        this->costFunction = RegularCost(dataConfig.nodeNum, dim, model, metricsType);  
        break;
    case CostTypeEnum::P:
        this->costFunction = PCost(dataConfig.nodeNum, dim, model, metricsType);  
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
    costFunction.calculate(this, cost, data);
    bestUpdate();
    swarmUpdate();
}

void Particle::predictCost(Flow *data, double *cost) {
    MetricsTypeEnum metricsTypes[MetricsNum] 
        = {MetricsTypeEnum::RMSE, MetricsTypeEnum::R2};
    costFunction.predict(Gbest, data, MetricsNum, metricsTypes, cost);

}

void Particle::costInitialize(Flow *data) {
    // cost initialization
    costFunction.calculate(this, cost, data);
    memcpy(Pbest_cost, cost, sizeof(double) * Npar);
    Gbest_cost = Pbest_cost[0];
    Gbest_id = 0;
}

void Particle::bestUpdate() {
    for (int p = 0; p < Npar; p++) {
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
    for (int p = 0; p < Npar; p++) {
        for (int d = 0; d < dim; d++) {
            if (Flow::tflow[d] == 0) {
                continue;
            }
            double sigma = abs(Gbest[d] - Par[p][d]);
            Par[p][d] = Gbest[d] + alpha * sigma * BoxMullerRandom();
            double rjump = 1.0 * random01();
            if(rjump<pjump) {
                Par[p][d]=1.0 * random01() *
                 (Xrandmax - Xrandmin) + Xrandmin;
            }

            if(Par[p][d] > Xmax) {
                Par[p][d] = Xmax;
            }
            else if(Par[p][d] < Xmin) {
                Par[p][d] = Xmin;
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
    for (int i = 0; i < Npar; i++) {
        delete [] Par[i];
        delete [] Pbest[i];
    }
    delete [] Par;
    delete [] Pbest;
}
