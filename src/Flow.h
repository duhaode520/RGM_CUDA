#ifndef DATARECORD_H
#define DATARECORD_H
#include <string>
#include "DataConfig.h"

class Flow {
public:
	int src;
	int dest;
	double flow;
	double dist;

	Flow() {};
	~Flow() ;

	static int* tflow ;  

	static void loadData(Flow* data, std::string filename);
};



#endif