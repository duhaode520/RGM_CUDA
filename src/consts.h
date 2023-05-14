#ifndef CONSTS_H
#define CONSTS_H
enum ModelTypeEnum {
    Reversed_Gravity,
    Reversed_Gravity_Exp
};

enum MetricsTypeEnum {
    RMSE,
    MAPE,
    R2
};

enum CostTypeEnum {
    Regular, 
    P
};

struct GlobalConfig {
    int nodeNum;
    int dim;
    CostTypeEnum costType;
    ModelTypeEnum modelType;
    MetricsTypeEnum metricsType;

    // for PCost
    int start_dim;
    int cDim;
};

const ModelTypeEnum MODEL_TYPE = ModelTypeEnum::Reversed_Gravity;
#endif