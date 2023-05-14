#include "Model.cuh"
#include "PSOConfig.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <sstream>
#include <assert.h>

__device__ __host__ Model* Model::create(ModelTypeEnum type, int nodeNum, int dim) {
    switch (type) {
    case ModelTypeEnum::Reversed_Gravity:
        return new RGM(nodeNum, dim);
    case ModelTypeEnum::Reversed_Gravity_Exp:
        return new RGM_EXP(nodeNum, dim);
    default:
        printf("Unknown model type\n");
        return nullptr;
    }
}

// void Model::destroy(Model* model) {
//     model->~Model();
//     cudaFree(model);
// }

RGM::RGM(int nodeNum, int dim) {
    this->_nodeNum = nodeNum;
    this->_dim = dim;
    _flowNum = (nodeNum - 1) * nodeNum;
    #if defined(__CUDA_ARCH__) 
        cudaMalloc(&_Push, sizeof(double) * nodeNum);
        cudaMalloc(&_Attr, sizeof(double) * nodeNum);
        cudaMalloc(&_beta, sizeof(double));
    #else
        cudaMallocManaged(&_Push, sizeof(double) * nodeNum);
        cudaMallocManaged(&_Attr, sizeof(double) * nodeNum);
        cudaMallocManaged(&_beta, sizeof(double));
    #endif // 
    
}


RGM::~RGM() {
    cudaFree(_Push);
    cudaFree(_Attr);
    cudaFree(_beta);
}

__device__ __host__ void RGM::_parse(int index, double* pars) {
    for(int c=0;c<_nodeNum;c++) {
        _Push[c]=pars[index*_dim+ c];
        _Attr[c]=pars[index*_dim+ _nodeNum + c];
    }
    *_beta = pars[index*_dim + _dim-1]/BETA_SCALE;
}

__device__ __host__ void RGM::pred(int index, double* pars, double* pred, FlowData* data) {
    // 从 particle 的维度中解析出需要的 Push Attr beta
    _parse(index, pars);
    // TODO: 这一步其实是可以用 CUDA 2D 的一些手段搞成并行的，但是我懒得学
    for (int i = 0; i < _flowNum; i++) {
        if (data[i].src > _flowNum) {
            printf("RGM::pred: flow %d is changed in kernel %d\n", i, index);
        }
        pred[i] = _FLOW_SCALE * _Push[data[i].src] * _Attr[data[i].dest] / pow(data[i].dist, *_beta);
    }
}

std::string RGM::getResult(double* pars) {
    _parse(0, pars);
    std::stringstream ss;
    // int extreme = -1;
    for (int i = 0; i < _nodeNum; i++) {
        ss << dataConfig->nodeNames[i] << "," << _Push[i] << "," << _Attr[i] << std::endl;
    }
    ss << "Beta:" <<*_beta << std::endl;
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

__device__ __host__ void RGM_EXP::pred(int index, double* pars, double* pred, FlowData* data) {
    // 从 particle 的维度中解析出需要的 Push Attr beta
    _parse(index, pars);

    for (int i = 0; i < _flowNum; i++) {
        // exp形式的距离衰减
        pred[i] = _FLOW_SCALE * _Push[data[i].src] * _Attr[data[i].dest] / exp(*_beta * data[i].dist);
    }
}