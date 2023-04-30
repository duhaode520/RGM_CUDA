#ifndef DATACONFIG_H
#define DATACONFIG_H
#include <iostream>
#include <map>
#include <fstream>

class DataConfig {
private:
    void parseConfig(std::ifstream& infile, std::map<std::string, std::string>& config);

    void parseList(std::string* resultList, std::string listStr);

public:

    int nodeNum; // number of nodes
    std::string dataType; // data file name
    std::string dataFile; // data file name
    int dim; // dimension of particles //TODO: Strange here
    int cDim; // dimesnion used in P-swarm
    int PSwarmNum; // number of P-swarm groups
    int rDim; // TODO: deprecated
    int flowNum; // number of flows
    std::string* nodeNames; // node names
    std::string outputFile; // outputfile
    int flowScale;
    int distScale;

    DataConfig();
    ~DataConfig();

    // load config from file
    void load(std::string filename);

};

extern DataConfig dataConfig;
#endif 

