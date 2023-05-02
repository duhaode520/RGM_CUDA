#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <iostream>
#include "DataConfig.h"

const double PI = acos(-1.0);

// Random number between [0, 1] with normal distribution
double BoxMullerRandom();

// Ramdom number beteween [0, 1]
double random01();

void parseArgs(int argc, char* argv[]);

#endif