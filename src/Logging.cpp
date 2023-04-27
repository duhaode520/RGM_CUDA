#include "Logging.h"
Logging logging;

void Logging::logStartInfo(DataConfig dataConfig) {
    
    clock_t tstart=clock();
    FILE* out;
    out=fopen((char*)logging.outputfile.data(),"w");
    time_t now = time(0); //当前系统时间
    char* dt = ctime(&now); //转换为字符串
    int DIMS=2*dataConfig.nodeNum+1;
    int K=int(ceil(1.0*DIMS/dataConfig.cDim));
    fprintf(out,"Time: %s\n",dt);
    fprintf(out,"Nodes:%d Flows:%d Npar:%d MaxIter:%d\n",dataConfig.nodeNum,dataConfig.flowNum,Particle::Npar,Particle::Maxiter);
    fprintf(out,"Xmin:%f Xmax:%f Xrandmin:%f Xrandmax:%f\n",Particle::Xmin,Particle::Xmax,Particle::Xrandmin,Particle::Xrandmax);
    fprintf(out,"alpha:%f pjump:%f K:%d CDIMS:%d RDIMS:%d ",Particle::alpha,Particle::pjump,K,dataConfig.cDim,dataConfig.rDim);
    fprintf(out,"flowscale:%d distscale:%d\n",dataConfig.flowScale,dataConfig.distScale);
    cout<<"Time: "<<dt<<endl;
    cout<<"Nodes: "<<dataConfig.nodeNum<<" Flows: "<<dataConfig.flowNum<<" Npar: "<<Particle::Npar<<" MaxIter: "<<Particle::Maxiter<<endl;
    cout<<"Xmin: "<<Particle::Xmin<<" Xmax: "<<Particle::Xmax<<" Xrandmin: "<<Particle::Xrandmin<<" Xrandmax:"<<Particle::Xrandmax<<endl;
    cout<<"alpha: "<<Particle::alpha<<" pjump: "<<Particle::pjump<<" K: "<<K<<" CDIMS: "<<dataConfig.cDim<<" RDIMS: "<<dataConfig.rDim<<endl;
    cout<<"flowscale: "<<dataConfig.flowScale<<" distscale: "<<dataConfig.distScale<<endl;
}