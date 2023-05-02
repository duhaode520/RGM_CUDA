#ifndef METRICS_H
#define METRICS_H
#include "Flow.h"
#include "consts.h"
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

    virtual __device__ double calc(Flow* data, double* pred, int size) = 0;

    static Metrics* createMetrics(MetricsTypeEnum type);

};

class RMSEMetric : public Metrics {
private:
    /* data */
public:
    __device__ double calc(Flow* data, double* pred, int size);
};

class RsquaredMetric : public Metrics {
public:
    __device__ double calc(Flow* data, double* pred, int size);
};

#endif