#include "Flow.h"
#include <iostream>
#include <fstream>
#include <cstring>
int* Flow::tflow = nullptr; // static member must be initialized outside the class

Flow::~Flow() {

}

void Flow::loadData(Flow* data, std::string filename) {
    std::cout << "Loading data from " << filename << std::endl;
    memset(tflow, 0, dataConfig->dim * sizeof(int));
    std::ifstream fdata(filename);
    if (!fdata.is_open()) {
        std::cout << "Error opening file " << filename << std::endl;
        return;
    }
    int i = 0;
    while (fdata >> data[i].src >> data[i].dest >> data[i].flow >> data[i].dist) {
        data[i].flow /= dataConfig->flowScale;
        data[i].dist /= dataConfig->distScale;
        tflow[data[i].src] += data[i].flow;
        tflow[dataConfig->nodeNum + data[i].dest] += data[i].flow;
        i++;
    }
    tflow[dataConfig->dim - 1] = 1; // set the last element to 1, which represents beta in RGM
    fdata.close();
}
