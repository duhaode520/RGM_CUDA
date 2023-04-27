#include "Logging.h"
Logging logging;

void Logging::logStartInfo(DataConfig dataConfig ,Particle particle) {
    
    clock_t tstart=clock();
    FILE* out;
    out=fopen((char*)logging.outputfile.data(),"w");
    time_t now = time(0); //当前系统时间
    char* dt = ctime(&now); //转换为字符串
    fprintf(out,"Time: %s\n",dt);
    fprintf(out,"Nodes:%d Records:%d Npar:%d MaxIter:%d\n",dataConfig.nodeNum,dataConfig.nRec,particle.Npar,particle.Maxiter);
    fprintf(out,"Xmin:%f Xmax:%f Xrandmin:%f Xrandmax:%f\n",Xmin,Xmax,Xrandmin,Xrandmax);
    fprintf(out,"alpha:%f pjump:%f K:%d CDIMS:%d RDIMS:%d scale:%d\n",alpha,pjump,K,CDIMS,RDIMS,SCALE);
    fprintf(out,"Betascale:%d beta:%f Threads_per_block:%d data:%s\n",BETA_SCALE,BETA,THREADS_PER_BLOCK,datafile);
    cout<<"Time: "<<dt<<endl;
    cout<<"Nodes: "<<NODES<<" Records: "<<NREC<<" Npar: "<<Npar<<" MaxIter: "<<Maxiter<<endl;
    cout<<"Xmin: "<<Xmin<<" Xmax: "<<Xmax<<" Xrandmin: "<<Xrandmin<<" Xrandmax:"<<Xrandmax<<endl;
    cout<<"alpha: "<<alpha<<" pjump: "<<pjump<<" K: "<<K<<" scale: "<<SCALE<<endl;
    cout<<"Threads_per_block: "<<THREADS_PER_BLOCK<<" data: "<<datafile<<endl;
}