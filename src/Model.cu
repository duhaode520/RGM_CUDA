#include "Model.cuh"
#include "PSOConfig.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <sstream>
#include <assert.h>
__device__ void checkCudaErrors(cudaError_t err, const char* file, const int line)
{
    if (err != cudaSuccess)
    {
        printf("CUDA error at %s:%d code=%d(%s) \n", file, line, err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__device__ __host__ Model* Model::create(ModelTypeEnum type, int nodeNum, int dim, int flowNum) {
    switch (type) {
    case ModelTypeEnum::Reversed_Gravity:
        return new RGM(nodeNum, dim, flowNum);
    case ModelTypeEnum::Reversed_Gravity_Exp:
        return new RGM_EXP(nodeNum, dim, flowNum);
    default:
        printf("Unknown model type\n");
        return nullptr;
    }
}

// void Model::destroy(Model* model) {
//     model->~Model();
//     cudaFree(model);
// }

RGM::RGM(int nodeNum, int dim, int flowNum) {
    this->_nodeNum = nodeNum;
    this->_dim = dim;
    this->_flowNum = flowNum;
    #if defined(__CUDA_ARCH__) 
        cudaMalloc(&_Push, sizeof(float) * nodeNum);
        cudaMalloc(&_Attr, sizeof(float) * nodeNum);
        cudaMalloc(&_beta, sizeof(float));
    #else
        cudaMallocManaged(&_Push, sizeof(float) * nodeNum);
        cudaMallocManaged(&_Attr, sizeof(float) * nodeNum);
        cudaMallocManaged(&_beta, sizeof(float));
    #endif // 
    
}


RGM::~RGM() {
    cudaFree(_Push);
    cudaFree(_Attr);
    cudaFree(_beta);
}

__device__ __host__ void RGM::_parse(int index, float* pars) {
    for(int c=0;c<_nodeNum;c++) {
        _Push[c]=pars[index*_dim+ c];
        _Attr[c]=pars[index*_dim+ _nodeNum + c];
    }
    *_beta = pars[index*_dim + _dim-1]/BETA_SCALE;
}

__device__ __host__ void RGM::pred(int index, float* pars, float* pred, FlowData* data) {
    // 从 particle 的维度中解析出需要的 Push Attr beta
    _parse(index, pars);
    // TODO: 这一步其实是可以用 CUDA 2D 的一些手段搞成并行的，但是我懒得学
    for (int i = 0; i < _flowNum; i++) {
        if (data[i].src > _flowNum) {
            printf("RGM::pred: flow %d is changed in kernel %d\n", i, index);
        }
        pred[i] = _FLOW_SCALE * _Push[data[i].src] * _Attr[data[i].dest] / powf(data[i].dist, *_beta);
        checkCudaErrors(cudaGetLastError(), __FILE__, __LINE__);
    }
}

std::string RGM::getResult(float* pars) {
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


__device__ __host__ void RGM_EXP::pred(int index, float* pars, float* pred, FlowData* data) {
    // 从 particle 的维度中解析出需要的 Push Attr beta
    _parse(index, pars);

    for (int i = 0; i < _flowNum; i++) {
        // exp形式的距离衰减
        pred[i] = _FLOW_SCALE * _Push[data[i].src] * _Attr[data[i].dest] / exp(*_beta * data[i].dist);
    }
}