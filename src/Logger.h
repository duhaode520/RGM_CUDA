#ifndef LOGGER_H
#define LOGGER_H
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


class Logger {
private:
    /* data */
    clock_t tstart;
    std::ofstream ofstream;
    clock_t tprev;
    void log_start(std::ostream& out, DataConfig dataConfig);
public:
    Logger(std::string outputfile);
    ~Logger();
    void logStartInfo(DataConfig dataConfig);
    void printSessionTime(std::string sessionName);
};


#endif 
