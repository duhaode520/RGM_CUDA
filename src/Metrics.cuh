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
private:
    /* data */
public:
    Metrics(/* args */) {}
    ~Metrics() {}

    virtual __device__ __host__ double calc(Flow* data, double* pred, int size) = 0;

    static void create(Metrics* metrics, MetricsTypeEnum type);

    static void destroy(Metrics* metrics);
};

class RMSEMetric : public Metrics {
private:
    /* data */
public:
    __device__ __host__ double calc(Flow* data, double* pred, int size) override;
};

class RsquaredMetric : public Metrics {
public:
    __device__ __host__ double calc(Flow* data, double* pred, int size) override;
};

#endif