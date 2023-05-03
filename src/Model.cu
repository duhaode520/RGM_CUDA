#include "Model.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <sstream>

Model* Model::createModel(ModelTypeEnum type, int nodeNum, int dim) {
    switch (type) {
        case ModelTypeEnum::Reversed_Gravity:
            return new RGM(nodeNum, dim);
        case ModelTypeEnum::Reversed_Gravity_Exp:
            return new RGM_EXP(nodeNum, dim);
        default:
            return nullptr;
    }
}

RGM::RGM(int nodeNum, int dim) {
    this->nodeNum = nodeNum;
    this->dim = dim;
    flowNum = (nodeNum - 1) * nodeNum / 2;
    Push = new double[nodeNum];
    Attr = new double[nodeNum];
    beta = 0;
}

RGM::~RGM() {
    delete[] Push;
    delete[] Attr;
}

__device__ __host__ void RGM::parse(int index, double* pars) {
    for(int c=0;c<nodeNum;c++) {
        Push[c]=pars[index*dim+ c];
        Attr[c]=pars[index*dim+ nodeNum + c];
    }
    beta = pars[index*dim + dim-1]/BETA_SCALE;
}

__device__ __host__ void RGM::pred(int index, double* pars, double* pred, Flow* data) {
    // 从 particle 的维度中解析出需要的 Push Attr beta
    parse(index, pars);
    // TODO: 这一步其实是可以用 CUDA 2D 的一些手段搞成并行的，但是我懒得学
    for (int i = 0; i < flowNum; i++) {
        int src = data[i].src;
        int dest = data[i].dest;
        double dist = data[i].dist;
        double gtFlow = data[i].flow;
        pred[i] = FLOW_SCALE * Push[src] * Attr[dest] / pow(dist, beta);
    }
}

std::string RGM::getResult(double* pars) {
    parse(0, pars);
    std::stringstream ss;
    // int extreme = -1;
    for (int i = 0; i < nodeNum; i++) {
        ss << dataConfig.nodeNames[i] << " " << Push[i] << " " << Attr[i] << std::endl;
    }
    ss << "Beta " << beta << std::endl;
    return ss.str();
}   

RGM_EXP::RGM_EXP(int nodeNum, int dim) : RGM(nodeNum, dim) {
}

__device__ __host__ void RGM_EXP::pred(int index, double* pars, double* pred, Flow* data) {
    // 从 particle 的维度中解析出需要的 Push Attr beta
    parse(index, pars);

    for (int i = 0; i < flowNum; i++) {
        int src = data[i].src;
        int dest = data[i].dest;
        double dist = data[i].dist;
        double gtFlow = data[i].flow;

        // exp形式的距离衰减
        pred[i] = FLOW_SCALE * Push[src] * Attr[dest] / exp(beta * dist);
    }
}