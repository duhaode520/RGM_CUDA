#ifndef MODEL_CUH
#define MODEL_CUH

#include "Flow.h"
#include "consts.h"
#include <cuda_runtime.h>

class Model {
protected:
    // parse the particle parameters
    virtual __device__ __host__ void parse(int index, double* par) = 0;

public:
    __device__ __host__ Model(){}
    __device__ __host__ virtual ~Model(){}

    virtual __device__ __host__ void pred(int index, double* par, double* pred, Flow* data) = 0;

    virtual std::string getResult(double* pars) = 0;

    virtual Model* prepareForDevice() = 0;

    virtual void leaveDevice() = 0;

    static __device__ __host__ Model* create(ModelTypeEnum type, int nodeNum, int dim);

    
//    static void destroy(Model* model);
};

class RGM : public Model {
protected:

    int nodeNum;
    int dim;
    int flowNum;
    
    double* Push;
    double* Attr;
    double* beta; // beta value for RGM, not an array

    static constexpr int BETA_SCALE = 10;
    static constexpr int FLOW_SCALE = 1;
    /**
     * @brief parse the particle parameters for RGM
     * the first 
     * 
     * @param par 
     * @return __device__ 
     */
    __device__ __host__ void parse(int index, double* pars) override;

public:
    __device__ __host__ RGM(int nodeNum, int dim);
    __device__ __host__ virtual ~RGM();

    __device__  __host__ void pred(int index, double* pars, double* pred, Flow* data) override;

    std::string getResult(double* pars);

    Model* prepareForDevice() ;
    
    void leaveDevice();
};

class RGM_EXP : public RGM {
protected:
    static constexpr int BETA_SCALE = 10;
    static constexpr int FLOW_SCALE = 1;
public:
    __device__ __host__ RGM_EXP(int nodeNum, int dim);

    __device__ __host__ void pred(int index, double* pars, double* pred, Flow* data) override;
};


#endif