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
#include "cuda_runtime.h"

using namespace std;

int main(int argc, char* argv[]) {
    dataConfig = new DataConfig();
    parseArgs(argc, argv);
    Logger logger(dataConfig->outputFile);
    logger.logStartInfo(dataConfig);

    srand(time(NULL));

    // cudaSetDevice(1);
    FlowData* data = new FlowData[dataConfig->flowNum];
    Flow::tflow = new int[dataConfig->dim];
    Flow::loadData(data, dataConfig->dataFile);   
    
    logger.printSessionTime("Data Loading");

    PParticle Ppar[dataConfig->PSwarmNum];
    Particle Qpar;
    PParticle::PGbest = new double[dataConfig->dim];

    GlobalConfig pConfig = {
        dataConfig->nodeNum,
        dataConfig->dim,
        CostTypeEnum::Regular,
        MODEL_TYPE,
        MetricsTypeEnum::RMSE,
        0,
        0
    };

    for (int i = 0; i < dataConfig->PSwarmNum; i++) {
        pConfig.start_dim = dataConfig->cDim * i;
        if (i == dataConfig->PSwarmNum - 1) {
            // 最后一个粒子的维度可能不是cDim
            int lastDim = dataConfig->dim - i * dataConfig->cDim;
            pConfig.cDim = lastDim;
            Ppar[i].initialize(pConfig);
        } else {
            pConfig.cDim = dataConfig->cDim;
            Ppar[i].initialize(pConfig);
        }
    }

    GlobalConfig qConfig = {
        dataConfig->nodeNum,
        dataConfig->dim,
        CostTypeEnum::Regular,
        MODEL_TYPE,
        MetricsTypeEnum::RMSE,
        0,
        0
    };

    Qpar.initialize(qConfig);

    logger.printSessionTime("Initialization");

    // cout << "PGbest:(Test) ";
    // for (int i = 0; i < dataConfig->dim; i++) {
    //     cout << PParticle::PGbest[i] << " ";
    // }
    // cout << endl;

    int Crossid;
    int iter = 0;
    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int s = 0; s < dataConfig->PSwarmNum; s++) {
            Ppar[s].train(data);
            logger.log("Iter:", iter, "PSwarm:", s, "P GbestCost:", Ppar[s].getGbestCost());
            Qpar.cooperate(&Ppar[s]);
        }
        
        Qpar.train(data);
        logger.log("Iter:", iter, "Q GbestCost:", Qpar.getGbestCost(), 
            "Gbest Beta:", Qpar.getGbestBeta()); 
        for (int s = 0; s < dataConfig->PSwarmNum; s++) {
            Ppar[s].cooperate(&Qpar);
        }

    }
    logger.printSessionTime("Training");

    #pragma region output
    logger.log(Qpar.getResult());

    double cost[Qpar.MetricsNum];
    Qpar.predictCost(data, cost);
    logger.log("Predict RMSE:", cost[0], "Predict R2:", cost[1]);

    #pragma endregion


    #pragma region release

    delete[] data;
    delete[] Flow::tflow;
    delete[] PParticle::PGbest;
    delete dataConfig;

    #pragma endregion
    return 0;
}