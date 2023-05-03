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

const ModelTypeEnum MODEL_TYPE = ModelTypeEnum::Reversed_Gravity;
#endif