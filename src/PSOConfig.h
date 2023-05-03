#ifndef PSOCONFIG_H
#define PSOCONFIG_H

// TODO: 这个文件可能之后会弃用，把对应的东西常量分别转移到文件中
const int N_PAR = 4096;// Number of particles each swarm
const int MAX_ITER = 3000;  //Maximum Iteration
const double X_MIN = 0.001; 
const double X_MAX = 100000000; 
const double X_RAND_MIN = 5;
const double X_RAND_MAX = 70;
const double ALPHA = 0.75;
const double P_JUMP = 0.001;
// const double SCALE = 1;
// const double BETA_SCALE = 10;
// const double MAX_BETA = 2;


#endif