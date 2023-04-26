/**
 * @file main.cpp
 * @author Haode Du
 * @brief main function of the program
 * @version 1
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <string>
#include <vector>

#include "PSOConfig.h"
#include "Flow.h"
#include "utils.h"
#include "Particle.h"

using namespace std;


/**
 * *注意: 在sample_code/mixed_test的结果显示
 * *编译时并不需要对代码做任何额外的改动即可自动混合C++和CUDA
 * *唯一的需要是在编译时nvcc编译.cu, g++编译.cpp
 */ 

int main(int argc, char* argv[]) {
    Flow datacache[dataConfig.flowNum];
    parseArgs(argc, argv);
    logStartInfo();
    Flow::loadData(dataConfig.dataFile);   
    Particle Ppar[Npar](dataConfig.cDim);
    Particle Qpar(dataConfig.dim);
    
    for (int i = 0; i < Npar; i++) {
        Ppar[i].initialize();
    }
    Qpar.initialize();

    double rjump,sigma;
    int Crossid;
    int iter = 0;
    int K; // number of P-swarms

    while (iter < Maxiter) {
        for (int s = 0; s < K; s++) {
            Ppar[s].train(datacache);
        }
        exchange();
        Qpar.train();
        exchange();
    }
    
    saveResults();
    release();
    return 0;
}