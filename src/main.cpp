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
#include "Logger.h"

using namespace std;


/**
 * *注意: 在sample_code/mixed_test的结果显示
 * *编译时并不需要对代码做任何额外的改动即可自动混合C++和CUDA
 * *唯一的需要是在编译时nvcc编译.cu, g++编译.cpp
 */ 

int main(int argc, char* argv[]) {
    parseArgs(argc, argv);
    Logger logger(dataConfig.outputFile);
    logger.logStartInfo(dataConfig);

    Flow* datacache = new Flow[dataConfig.flowNum];
    Flow::tflow = new int[dataConfig.dim];
    Flow::loadData(datacache, dataConfig.dataFile);   
    logger.printSessionTime("Data Loading");

    Particle Ppar[dataConfig.PSwarmNum];
    Particle Qpar;

    for (int i = 0; i < dataConfig.PSwarmNum; i++) {
        if (i == dataConfig.PSwarmNum - 1) {
            // 最后一个粒子的维度可能不是cDim
            int lastDim = dataConfig.dim - i * dataConfig.cDim;
            Ppar[i].initialize(lastDim);
        } else {
            Ppar[i].initialize(dataConfig.cDim);
        }
        Ppar[i].setCost(CostTypeEnum::P,
            ModelTypeEnum::Reversed_Gravity, MetricsTypeEnum::RMSE);
    }
    Qpar.initialize(dataConfig.dim);
    Qpar.setCost(CostTypeEnum::Regular,
        ModelTypeEnum::Reversed_Gravity, MetricsTypeEnum::RMSE);

    logger.printSessionTime("Initialization");

    int Crossid;
    int iter = 0;
    for (int iter = 0; iter < Maxiter; iter++) {
        for (int s = 0; s < dataConfig.PSwarmNum; s++) {
            Ppar[s].train(datacache);
            logger.log("Iter:", iter, "PSwarm:", s, "P GbestCost:", Ppar[s].getGbestCost());
        }
        exchange();
        Qpar.train(datacache);
        logger.log("Iter:", iter, "Q GbestCost:", Qpar.getGbestCost(), 
            "Gbest Beta:", Qpar.getGbestBeta()); 
        exchange();
    }
    
    saveResults();
    #pragma region release

    delete[] datacache;
    delete[] Flow::tflow;

    #pragma endregion
    return 0;
}