#include "stars-pm-cuda.h"

#include "stars-pm-cuda.h"




__global__ void k_updatepos( float *pos, float *vel ) {
  int tx=threadIdx.x;
  int bx=blockIdx.x;


  int gIdx = bx*NPBLOCK + tx;

  float lpos = pos[gIdx];

  lpos = lpos + vel[gIdx] * DT;
  lpos = lpos + LBOX*((lpos < 0) - (lpos > LBOX));

  pos[gIdx] = lpos;


}



void updatepos(coord pos[NP][NP][NP],
               coord vel[NP][NP][NP]) {
  dim3 dimGriddata(3*NPART/NPBLOCK);
  dim3 dimBlockdata(NPBLOCK);
  P4A_launch_kernel(dimGriddata,dimBlockdata,k_updatepos,(float *)pos,(float *)vel);
}

