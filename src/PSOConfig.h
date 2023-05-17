#ifndef PSOCONFIG_H
#define PSOCONFIG_H

// TODO: 这个文件可能之后会弃用，把对应的东西常量分别转移到文件中
// TODO: 这个文件里的常量可能需要未来保存到 ini 中，然后编写一个loader来读取，现在每一次更改需要重新编译，非常灾难

/** HINT: For now, the N_PAR is limited in 2048. 
 * Bigger N_PAR will cause CUDA_EXCEPTION 5 for unknown reason.
*/
const int N_PAR = 512;// Number of particles each swarm 

const int MAX_ITER = 10;  //Maximum Iteration
const double X_MIN = 0.001; 
const double X_MAX = 100000000; 
const double X_RAND_MIN = 5;
const double X_RAND_MAX = 70;
const double ALPHA = 0.75;
const double P_JUMP = 0.001;
// const double SCALE = 1;
const double BETA_SCALE = 10;
const double MAX_BETA = 2;


#endif