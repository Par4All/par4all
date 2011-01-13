#include "stars-pm-cuda.h"



__global__ void k_updatepos( float *pos, float *vel ) {
  int tx=threadIdx.x;
  int bx=blockIdx.x;

  __shared__ float spos[NPBLOCK][3];
  __shared__ float svel[NPBLOCK][3];


  // Charge les positions depuis la mémoire globale vers la mémoire partagée
  spos[tx][0]=pos[(bx*NPBLOCK+tx)*3+0];
  spos[tx][1]=pos[(bx*NPBLOCK+tx)*3+1];
  spos[tx][2]=pos[(bx*NPBLOCK+tx)*3+2];

  // Idem vélocité
  svel[tx][0]=vel[(bx*NPBLOCK+tx)*3+0];
  svel[tx][1]=vel[(bx*NPBLOCK+tx)*3+1];
  svel[tx][2]=vel[(bx*NPBLOCK+tx)*3+2];

  __syncthreads(); // FIXME inutile


  spos[tx][0]+=svel[tx][0]*DT;
  spos[tx][1]+=svel[tx][1]*DT;
  spos[tx][2]+=svel[tx][2]*DT;

  pos[(bx*NPBLOCK+tx)*3+0]=spos[tx][0]+LBOX*((spos[tx][0]<0)-(spos[tx][0]>LBOX));
  pos[(bx*NPBLOCK+tx)*3+1]=spos[tx][1]+LBOX*((spos[tx][1]<0)-(spos[tx][1]>LBOX));
  pos[(bx*NPBLOCK+tx)*3+2]=spos[tx][2]+LBOX*((spos[tx][2]<0)-(spos[tx][2]>LBOX));


}

void updatepos(coord pos[NP][NP][NP],
               coord vel[NP][NP][NP]) {
  dim3 dimGriddata(NP/NPBLOCK);
  dim3 dimBlockdata(NPBLOCK);
  P4A_launch_kernel(dimGriddata,dimBlockdata,k_updatepos,(float *)pos,(float *)vel);
}

