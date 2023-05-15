#include "Metrics.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ __host__ Metrics* Metrics::create(MetricsTypeEnum type) {
    switch (type) {
    case MetricsTypeEnum::RMSE:
        return new RMSEMetric();
    case MetricsTypeEnum::R2:
        return new RsquaredMetric();
    default:
        printf("Error: unknown metrics type\n");
        return nullptr;
    }
}

Metrics* Metrics::prepareForDevice() {
    Metrics* deviceMetrics;
    cudaMalloc((void**)&deviceMetrics, sizeof(Metrics));
    cudaMemcpy(deviceMetrics, this, sizeof(Metrics), cudaMemcpyHostToDevice);
    return deviceMetrics;
}

void Metrics::leaveDevice() {
    // temporarily do nothing
}

// void Metrics::destroy(Metrics* metrics) {
//     metrics->~Metrics();
//     cudaFree(metrics);
// }

__device__ __host__ float RMSEMetric::calc(FlowData* data, float* pred, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += (data[i].flow - pred[i]) * (data[i].flow - pred[i]);
    }
    return sqrt(sum / size);
}

__device__ __host__ float RsquaredMetric::calc(FlowData* data, float* pred, int size) {
    float mean = 0;
    for (int i = 0; i < size; i++) {
        mean += data[i].flow;
    }
    mean /= size;

    float ss_tot = 0;
    float ss_res = 0;
    for (int i = 0; i < size; i++) {
        ss_tot += (data[i].flow - mean) * (data[i].flow - mean);
        ss_res += (data[i].flow - pred[i]) * (data[i].flow - pred[i]);
    }
    return 1 - ss_res / ss_tot;
}

