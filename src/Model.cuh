#ifndef MODEL_CUH
#define MODEL_CUH

#include "Flow.h"
#include "consts.h"

class Model {
private:
    // parse the particle parameters
    virtual __device__ void parse(int index, double* par);
public:
    Model(){}
    ~Model(){}

   virtual __device__ void pred(int index, double* par, double* pred, Flow* data);

   static Model* createModel(ModelTypeEnum type);
};

class RGM : public Model {
protected:

    int nodeNum;
    int dim;
    int flowNum;
    
    double* Push;
    double* Attr;
    double beta;

    static const int BETA_SCALE = 10;
    static const int FLOW_SCALE = 1;
    /**
     * @brief parse the particle parameters for RGM
     * the first 
     * 
     * @param par 
     * @return __device__ 
     */
    __device__ void parse(int index, double* pars);
public:
    RGM(int nodeNum, int dim);
    ~RGM();
    __device__ void pred(int index, double* pars, double* pred, Flow* data);
};

class RGM_EXP : public RGM {
protected:
    static const int BETA_SCALE = 10;
    static const int FLOW_SCALE = 1;
public:

    __device__ void pred(int index, double* pars, double* pred, Flow* data);
};


#endif