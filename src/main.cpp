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
#include "Model.cuh"

using namespace std;

int main(int argc, char* argv[]) {
    dataConfig = new DataConfig();
    parseArgs(argc, argv);
    Logger logger(dataConfig->outputFile);
    logger.logStartInfo(dataConfig);

    Flow* datacache = new Flow[dataConfig->flowNum];
    Flow::tflow = new int[dataConfig->dim];
    Flow::loadData(datacache, dataConfig->dataFile);   
    logger.printSessionTime("Data Loading");

    Particle Ppar[dataConfig->PSwarmNum];
    Particle Qpar;

    for (int i = 0; i < dataConfig->PSwarmNum; i++) {
        if (i == dataConfig->PSwarmNum - 1) {
            // 最后一个粒子的维度可能不是cDim
            int lastDim = dataConfig->dim - i * dataConfig->cDim;
            Ppar[i].initialize(lastDim);
        } else {
            Ppar[i].initialize(dataConfig->cDim);
        }
        Ppar[i].setCost(CostTypeEnum::P, MetricsTypeEnum::RMSE);
        Ppar[i].setModel(MODEL_TYPE);
    }
    Qpar.initialize(dataConfig->dim);
    Qpar.setCost(CostTypeEnum::Regular, MetricsTypeEnum::RMSE);
    Qpar.setModel(MODEL_TYPE);

    logger.printSessionTime("Initialization");

    int Crossid;
    int iter = 0;
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // for (int s = 0; s < dataConfig->PSwarmNum; s++) {
        //     Ppar[s].train(datacache);
        //     logger.log("Iter:", iter, "PSwarm:", s, "P GbestCost:", Ppar[s].getGbestCost());
        // }
        // exchange();
        Qpar.train(datacache);
        logger.log("Iter:", iter, "Q GbestCost:", Qpar.getGbestCost(), 
            "Gbest Beta:", Qpar.getGbestBeta()); 
        // exchange();
    }
    logger.printSessionTime("Training");

    #pragma region output
    logger.log(Qpar.getResult());

    double cost[Qpar.MetricsNum];
    Qpar.predictCost(datacache, cost);
    logger.log("Predict RMSE:", cost[0], "Predict R2:", cost[1]);

    #pragma endregion


    #pragma region release

    delete[] datacache;
    delete[] Flow::tflow;
    delete dataConfig;
    #pragma endregion
    return 0;
}