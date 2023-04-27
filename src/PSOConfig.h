
#ifndef PSOCONFIG_H
#define PSOCONFIG_H

// TODO: 这个文件可能之后会弃用，把对应的东西常量分别转移到文件中
const int Npar = 4096;// Number of particles each swarm
const int Maxiter = 3000;  //Maximum Iteration
const double Xmin = 0.001; 
const double Xmax = 100000000; 
const double Xrandmin = 5;
const double Xrandmax = 70;
const double alpha = 0.75;
const double pjump = 0.001;
const double SCALE = 1;
const double BETA_SCALE = 10;
const double MAX_BETA = 2;

#endif