#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <iostream>
#include "DataConfig.h"

const float PI = acos(-1.0);

// Random number between [0, 1] with normal distribution
float BoxMullerRandom();

// Ramdom number beteween [0, 1]
float random01();

void parseArgs(int argc, char* argv[]);


#endif