#ifndef MODEL_CUH
#define MODEL_CUH

#include "Flow.h"
#include "consts.h"
#include <cuda_runtime.h>

class Model {
protected:
    // parse the particle parameters
    virtual __device__ __host__ void _parse(int index, float* par) = 0;

public:
    __device__ __host__ Model(){}
    __device__ __host__ virtual ~Model(){}

    virtual __device__ __host__ void pred(int index, float* par, float* pred, FlowData* data) = 0;

    virtual std::string getResult(float* pars) = 0;

    virtual Model* prepareForDevice() = 0;

    virtual void leaveDevice() = 0;

    static __device__ __host__ Model* create(ModelTypeEnum type, int nodeNum, int dim, int flowNum);

    
//    static void destroy(Model* model);
};

class RGM : public Model {
protected:

    int _nodeNum;
    int _dim;
    int _flowNum;
    
    float* _Push;
    float* _Attr;
    float* _beta; // beta value for RGM, not an array

    static constexpr int _FLOW_SCALE = 1;
    /**
     * @brief parse the particle parameters for RGM
     * the first 
     * 
     * @param par 
     * @return __device__ 
     */
    __device__ __host__ void _parse(int index, float* pars) override;

public:
    __device__ __host__ RGM(int nodeNum, int dim, int flowNum);
    __device__ __host__ virtual ~RGM();

    __device__  __host__ void pred(int index, float* pars, float* pred, FlowData* data) override;

    std::string getResult(float* pars);

    Model* prepareForDevice() ;
    
    void leaveDevice();
};

class RGM_EXP : public RGM {
protected:
    static constexpr int _FLOW_SCALE = 1;
public:
    __device__ __host__ RGM_EXP(int nodeNum, int dim, int flowNum) : RGM(nodeNum, dim, flowNum) {};

    __device__ __host__  void pred(int index, float* pars, float* pred, FlowData* data) override;
};


#endif