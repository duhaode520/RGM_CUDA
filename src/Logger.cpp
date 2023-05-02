#include "Logger.h"

void Logger::log_start(std::ostream &out, DataConfig dataConfig) {
    time_t now = time(0);   // 当前系统时间
    char *dt = ctime(&now); // 转换为字符串

    out << "Time: " << dt << std::endl;
    out << "Nodes: " << dataConfig.nodeNum << " Flows: " << dataConfig.flowNum 
        << " Npar: " << Particle::Npar << " MaxIter: " << Particle::Maxiter << std::endl;
    out << "Xmin: " << Particle::Xmin << " Xmax: " << Particle::Xmax 
        << " Xrandmin: " << Particle::Xrandmin << " Xrandmax:" << Particle::Xrandmax << std::endl;
    out << "alpha: " << Particle::alpha << " pjump: " << Particle::pjump 
        << " Number of P-Swarm groups: " << dataConfig.PSwarmNum << " CDIMS: " << dataConfig.cDim << " RDIMS: " << dataConfig.rDim << std::endl;
    out << "flowscale: " << dataConfig.flowScale << " distscale: " << dataConfig.distScale << std::endl;
}

Logger::Logger(std::string outputfile)
{
    ofstream.open(outputfile);
    tstart = clock();
    tprev = tstart;
}

Logger::~Logger() {
    ofstream.close();
}

void Logger::logStartInfo(DataConfig dataConfig) {
    log_start(std::cout, dataConfig);
    log_start(ofstream, dataConfig);
}

void Logger::printSessionTime(std::string sessionName) {
    clock_t tend = clock();
    double time = (tend - tprev) / (double)CLOCKS_PER_SEC;
    std::cout << sessionName << " finished in time: " << time << "s" << std::endl;
    ofstream << sessionName << " finished in time: " << time << "s" << std::endl;
    tprev = tend;
}

template <typename T, typename... Args>
void Logger::logItem(T arg, Args... args) {
    std::cout << arg << " ";
    ofstream << arg << " ";
    log(args...);
}

void Logger::logItem() {
    std::cout << std::endl;
    ofstream << std::endl;
}

template <typename... Args>
void Logger::log(Args... args) {
    logItem(args...);
}
