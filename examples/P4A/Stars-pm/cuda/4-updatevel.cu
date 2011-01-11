#include <math.h>

#include "stars-pm-cuda.h"


__global__ void k_updatevel( float *vel, float *force, int *data, int coord, float dt )
{
  int tx=threadIdx.x;
  int bx=blockIdx.x;

  // FIXME coalescence
  vel[ (bx * NPBLOCK + tx) * 3 + coord ] += force[ data [ bx*NPBLOCK + tx ] ] * dt;
}


void updatevel(coord vel[NP][NP][NP],
               float force[NP][NP][NP],
               int data[NP][NP][NP],
               int coord,
               float dt) {
  dim3 dimGriddata(NP/NPBLOCK);
  dim3 dimBlockdata(NPBLOCK);
  P4A_launch_kernel(dimGriddata,dimBlockdata,k_updatevel,(float *)vel, (float *)force, (int *)data, coord, dt);
}

