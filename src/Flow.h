#ifndef DATARECORD_H
#define DATARECORD_H
#include <string>
#include "DataConfig.h"

struct FlowData {
	unsigned int src;
	unsigned int dest;
	double flow;
	double dist;
};

namespace Flow {
	extern int* tflow ;  

	void loadData(FlowData* data, std::string filename);
};



#endif