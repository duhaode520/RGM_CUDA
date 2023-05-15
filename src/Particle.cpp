#include "Particle.h"
#include <cstring>

void Particle::_particle_init(int dim) {
    this->_par_dim = dim;

    // 1. particles initialization
    _Par = new float*[N_PAR];
    _Pbest = new float*[N_PAR];
    for (int i = 0; i < N_PAR; i++) {
        _Par[i] = new float[_par_dim];
        _Pbest[i] = new float[_par_dim];
    }

    for (int i = 0; i < N_PAR; i++) {
        for (int d = 0; d < _par_dim; d++) {
            if (Flow::tflow[d] > 0) {
                _Par[i][d] = X_RAND_MIN + (X_RAND_MAX - X_RAND_MIN) * rand() / RAND_MAX;
            } else {
                _Par[i][d] = 0;
            }
            _Pbest[i][d] = _Par[i][d];
        }
    }

    // global best should be complete, not just focus on partial dimensions
    _Gbest = new float[_par_dim];
    for (int d = 0; d < _par_dim; d++) {
        _Gbest[d] = _Pbest[0][d];
    }

}

void Particle::_setCost(CostTypeEnum costType, MetricsTypeEnum metricsType) {
    _cost_func = Cost::create(costType, _config.nodeNum, _config.dim, _config.flowNum, _model, metricsType);
}

void Particle::_setModel(ModelTypeEnum modelType) {
    this->_model = Model::create(modelType, _config.nodeNum, _config.dim, _config.flowNum);
}

Particle::~Particle() {
    delete [] _Gbest;
    for (int i = 0; i < N_PAR; i++) {
        delete [] _Par[i];
        delete [] _Pbest[i];
    }
    delete [] _Par;
    delete [] _Pbest;

    delete _model;
    delete _cost_func;
}

void Particle::initialize(GlobalConfig config) {
    this->_config = config;
    _particle_init(config.dim);
    _setModel(config.modelType);
    _setCost(config.costType, config.metricsType);
}

void Particle::train(FlowData *data)
{
    if (!_cost_init) {
        _cost_init = true;
        _costInitialize(data);
    }
    _costCalc(data);
    _bestUpdate();
    _swarmUpdate();
}

void Particle::predictCost(FlowData *data, float *cost) {
    MetricsTypeEnum metricsTypes[MetricsNum] 
        = {MetricsTypeEnum::RMSE, MetricsTypeEnum::R2};
    _cost_func->predict(_Gbest, data, MetricsNum, metricsTypes, cost);

}

void Particle::_costInitialize(FlowData *data) {
    // cost initialization
    _costCalc(data);
    memcpy(_Pbest_cost, _cost, sizeof(float) * N_PAR);
    _Gbest_cost = _Pbest_cost[0];
    _Gbest_id = 0;
}

void Particle::_costCalc(FlowData *data) {
    _cost_func->calculate(_config, _Par, N_PAR, data, _cost);
}

void Particle::_bestUpdate() {
    // std::cout << "Cost: ";
    for (int p = 0; p < N_PAR; p++) {
        float cur_cost = _cost[p];
        // std::cout << cur_cost << " ";
        if (cur_cost < _Pbest_cost[p]) {
            for (int d = 0; d < _par_dim; d++) {
                _Pbest[p][d] = _Par[p][d];
            }
            _Pbest_cost[p] = cur_cost;
        }

        if (cur_cost < _Gbest_cost) {
            for (int d = 0; d < _par_dim; d++) {
                _Gbest[d] = _Par[p][d];
            }
            _Gbest_cost = cur_cost;
            _Gbest_id = p;
        }
    }
    // std::cout << std::endl;
}

void Particle::_swarmUpdate() {
    // TODO: 未来如果要引入不同的 update 方式可以把这个函数改成类
    // TODO: 这一部分可以用CUDA加速
    for (int p = 0; p < N_PAR; p++) {
        for (int d = 0; d < _par_dim; d++) {
            if (Flow::tflow[d] == 0) {
                continue;
            }
            float sigma = fabs(_Gbest[d] - _Par[p][d]);
            _Par[p][d] = _Gbest[d] + ALPHA * sigma * BoxMullerRandom();
            float rjump = 1.0 * random01();
            if(rjump<P_JUMP) {
                _Par[p][d]=1.0 * random01() *
                 (X_RAND_MAX - X_RAND_MIN) + X_RAND_MIN;
            }

            if(_Par[p][d] > X_MAX) {
                _Par[p][d] = X_MAX;
            }
            else if(_Par[p][d] < X_MIN) {
                _Par[p][d] = X_MIN;
            }
        }
    }
}

float Particle::getGbestCost() {
    return _Gbest_cost;
}

float Particle::getGbestBeta() {
    return _Gbest[_par_dim - 1] / BETA_SCALE;
}

std::string Particle::getResult() { 
    return _model->getResult(_Gbest);
}

void Particle::cooperate(Particle *other) {
    int crossID = rand() % (N_PAR / 2);
    while(crossID == other->_Gbest_id) {
        crossID = rand() % (N_PAR / 2);
    }
    int minDim = (_par_dim < other->_par_dim) ? _par_dim : other->_par_dim;
    for (int d = 0; d < minDim; d++) {
        _Par[crossID][other->_config.start_dim + d] 
            = other->_Gbest[_config.start_dim + d];
    }
}

float* PParticle::PGbest = nullptr;

void PParticle::__copyToPGbest() {
    for (int d = 0; d < _par_dim; d++) {
        PGbest[d + _config.start_dim] = _Gbest[d];
    }
}

void PParticle::_costCalc(FlowData *data)
{
    // generate full parameter and copy the global best
    float** fullPar = new float*[N_PAR];
    for (int i = 0; i < N_PAR; i++) {
        fullPar[i] = new float[_config.dim];
        memcpy(fullPar[i], PGbest, sizeof(float) * _config.dim);
    }

    // set the focused dimension as the value in particles
    for (int i = 0; i < N_PAR; i++) {
        for (int d = 0; d < _config.dim; d++) {
            fullPar[i][d] = _Par[i][d + _config.start_dim];
        }
    }
    
    _cost_func->calculate(_config, _Par, N_PAR, data, _cost);

    // release
    for (int i = 0; i < N_PAR; i++) {
        delete [] fullPar[i];
    }
    delete [] fullPar;
}

void PParticle::_bestUpdate() {
    Particle::_bestUpdate();
    // update PGbest
    __copyToPGbest();
}

void PParticle::initialize(GlobalConfig config)
{
    this->_config = config;
    _particle_init(config.cDim);
    _setModel(config.modelType);
    _setCost(config.costType, config.metricsType);
    
    // Set PGbest
    __copyToPGbest();
}
