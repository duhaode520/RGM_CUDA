#include "Model.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <sstream>

Model* Model::create(ModelTypeEnum type, int nodeNum, int dim) {
    switch (type) {
    case ModelTypeEnum::Reversed_Gravity:
        return new RGM(nodeNum, dim);
    case ModelTypeEnum::Reversed_Gravity_Exp:
        return new RGM_EXP(nodeNum, dim);
    default:
        throw std::runtime_error("Unknown model type");
    }
}

// void Model::destroy(Model* model) {
//     model->~Model();
//     cudaFree(model);
// }

RGM::RGM(int nodeNum, int dim) {
    this->nodeNum = nodeNum;
    this->dim = dim;
    flowNum = (nodeNum - 1) * nodeNum / 2;
    cudaMallocManaged(&Push, sizeof(double) * nodeNum);
    cudaMallocManaged(&Attr, sizeof(double) * nodeNum);
    cudaMallocManaged(&beta, sizeof(double));
}

RGM::~RGM() {
    cudaFree(Push);
    cudaFree(Attr);
    cudaFree(beta);
}

__device__ __host__ void RGM::parse(int index, double* pars) {
    for(int c=0;c<nodeNum;c++) {
        Push[c]=pars[index*dim+ c];
        Attr[c]=pars[index*dim+ nodeNum + c];
    }
    *beta = pars[index*dim + dim-1]/BETA_SCALE;
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
        pred[i] = FLOW_SCALE * Push[src] * Attr[dest] / pow(dist, *beta);
    }
}

std::string RGM::getResult(double* pars) {
    parse(0, pars);
    std::stringstream ss;
    // int extreme = -1;
    for (int i = 0; i < nodeNum; i++) {
        ss << dataConfig->nodeNames[i] << " " << Push[i] << " " << Attr[i] << std::endl;
    }
    ss << "Beta " <<*beta << std::endl;
    return ss.str();
}   

Model* RGM::prepareForDevice() {
    RGM* d_model;
    cudaMalloc((void**)&d_model, sizeof(RGM));
    cudaMemcpy(d_model, this, sizeof(RGM), cudaMemcpyHostToDevice);
    // 这里目前 RGM 有的剩下的几个指针项 Push Attr Beta
    // 是用 cudaMallocManaged 分配的，所以不需要再次拷贝
    return d_model;
} 

void RGM::leaveDevice() {
    // temporarily do nothing.
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
        pred[i] = FLOW_SCALE * Push[src] * Attr[dest] / exp(*beta * dist);
    }
}