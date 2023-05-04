#include "Metrics.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void Metrics::create(Metrics* metrics, MetricsTypeEnum type) {
    switch (type) {
    case MetricsTypeEnum::RMSE:
        cudaMallocManaged((void**)&metrics, sizeof(RMSEMetric));
        new (metrics) RMSEMetric();
        break;
    case MetricsTypeEnum::R2:
        cudaMallocManaged((void**)&metrics, sizeof(RsquaredMetric));
        new (metrics) RsquaredMetric();
        break;
    default:
        throw std::runtime_error("Unknown metrics type");
    }
}

void Metrics::destroy(Metrics* metrics) {
    metrics->~Metrics();
    cudaFree(metrics);
}

__device__ __host__ double RMSEMetric::calc(Flow* data, double* pred, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += (data[i].flow - pred[i]) * (data[i].flow - pred[i]);
    }
    return sqrt(sum / size);
}

__device__ __host__ double RsquaredMetric::calc(Flow* data, double* pred, int size) {
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

