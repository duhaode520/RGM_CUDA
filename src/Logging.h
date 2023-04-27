#ifndef LOGGING_H
#define LOGGING_H
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <string>
#include <vector>
#include "DataConfig.h"
#include "Particle.h"

using namespace std;
class Logging {
private:
    /* data */
    std::string outputfile;
public:
    void Logging::logStartInfo(DataConfig dataConfig);
};


#endif 

