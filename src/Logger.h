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
    
    template <typename T, typename... Args>  // 所有模板函数写在头文件里才能找见
    void logItem(T arg, Args... args) {
        std::cout << arg << " ";
        ofstream << arg << " ";
        log(args...);
    }

    void logItem() {
        std::cout << std::endl;
        ofstream << std::endl;
    }

public:
    Logger(std::string outputfile);
    ~Logger();
    void logStartInfo(DataConfig dataConfig);
    void printSessionTime(std::string sessionName);

    template <typename... Args>
    void log(Args... args) {
        logItem(args...);
    }

    
};


#endif

