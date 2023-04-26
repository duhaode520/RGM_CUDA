#ifndef PARTICLE_H
#define PARTICLE_H
#include "utils.h"
#include "Flow.h"
#include "Cost.cuh"

class Particle {
private:
    /* data */
    // int Npar; // number of particles
    int dim; // dimension of particles
    double** Par;
    double** Pbest;
    double* Lpar;
    double* cost;

    double* Gbest;
    double* Gbest_cost;
    Cost costFunction;

    static const double Xmin = 0.001;
    static const double Xmax = 100000000;
    static const double Xrandmin = 5;
    static const double Xrandmax = 70;
    static const int Npar = 4096;// Number of particles each swarm


public:
    Particle(int dim);

    // 初始化 Pbest Par Gbest
    void initialize();

    void setCost(Cost costFunction);
    // 训练
    void train(Flow* data);

    ~Particle();
};


#endif