#include "Particle.h"

Particle::Particle(int dim) {
    this->dim = dim;
    Par = new double*[Npar];
    Pbest = new double*[Npar];
    for (int i = 0; i < Npar; i++) {
        Par[i] = new double[dim];
        Pbest[i] = new double[dim];
    }

}

void Particle::initialize() {
    // sample code
    for (int i = 0; i < Npar; i++) {
        for (int j = 0; j < dim; j++) {
            Par[i][j] = Xrandmin + (Xrandmax - Xrandmin) * rand() / RAND_MAX;
            Pbest[i][j] = Par[i][j];
        }
    }

}

void Particle::train(Flow* data) {
    costFunction.calcuate(this, cost, data);
}

Particle::~Particle() {
}
