#ifndef DATARECORD_H
#define DATARECORD_H
#include <string>

class Flow {
private:
	/* data */
public:
	int src;
	int dest;
	int flow;
	float dist;

	Flow(int src, int dest, int flow, float dist);
	~Flow();

	static int tflow[dataConfig.dim]; // TODO: 这里和dataConfig不应该有这么深的耦合

	static void loadData(std::string filename);
};



#endif