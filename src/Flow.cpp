#include "Flow.h"
#include <iostream>
#include <fstream>
Flow::~Flow() {

}

void Flow::loadData(Flow* data, std::string filename) {
    std::ifstream fdata(filename);
    if (!fdata.is_open()) {
        std::cout << "Error opening file " << filename << std::endl;
        return;
    }
    int i = 0;
    while (fdata >> data[i].src >> data[i].dest >> data[i].flow >> data[i].dist) {
        i++;
        tflow[data[i].src] += data[i].flow;
        tflow[dataConfig.nodeNum + data[i].dest] += data[i].flow;
    }
    tflow[dataConfig.dim - 1] = 1; // set the last element to 1, which represents beta in RGM
    fdata.close();
}
