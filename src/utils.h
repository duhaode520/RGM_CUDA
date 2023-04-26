#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <iostream>
#include "DataConfig.h"

const double PI = acos(-1.0);

// Random number generator
double BoxMullerRandom();

void parseArgs(int argc, char* argv[]);

#endif