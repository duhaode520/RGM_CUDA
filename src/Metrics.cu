#include "Metrics.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

Metrics* Metrics::createMetrics(MetricsTypeEnum type) {
    switch (type) {
    case MetricsTypeEnum::RMSE:
        return new RMSEMetric();
    case MetricsTypeEnum::R2:
        return new RsquaredMetric();
    default:
        return nullptr;
    }
}

__device__ double RMSEMetric::calc(Flow* data, double* pred, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += (data[i].flow - pred[i]) * (data[i].flow - pred[i]);
    }
    return sqrt(sum / size);
}

__device__ double RsquaredMetric::calc(Flow* data, double* pred, int size) {
    double mean = 0;
    for (int i = 0; i < size; i++) {
        mean += data[i].flow;
    }
    mean /= size;

    double ss_tot = 0;
    double ss_res = 0;
    for (int i = 0; i < size; i++) {
        ss_tot += (data[i].flow - mean) * (data[i].flow - mean);
        ss_res += (data[i].flow - pred[i]) * (data[i].flow - pred[i]);
    }
    return 1 - ss_res / ss_tot;

}

