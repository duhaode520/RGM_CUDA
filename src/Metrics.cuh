#ifndef METRICS_H
#define METRICS_H
#include "Flow.h"
#include "consts.h"
#include <cuda_runtime.h>

/**
 * @brief Virtual class for metrics used in cost function and evaluation
 * 
 */
class Metrics {
protected:
    /* data */
public:
    __device__ __host__ Metrics(/* args */) {}
    __device__ __host__ virtual ~Metrics() {}

    virtual __device__ __host__  float calc(FlowData* data, float* pred, int size) = 0;

    static __device__ __host__ Metrics* create(MetricsTypeEnum type);

    virtual Metrics* prepareForDevice();

    void leaveDevice();
    // static void destroy(Metrics* metrics);
};

class RMSEMetric : public Metrics {
private:
    /* data */
public:
    __device__ __host__  float calc(FlowData* data, float* pred, int size) override;
};

class RsquaredMetric : public Metrics {
public:
    __device__ __host__  float calc(FlowData* data, float* pred, int size) override;
};

#endif