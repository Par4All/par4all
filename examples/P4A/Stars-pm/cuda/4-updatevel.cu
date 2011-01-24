#include <math.h>

#include "stars-pm-cuda.h"


__global__ void k_updatevel( float *vel, float *force, int *data, int coord, float dt )
{
  int tx=threadIdx.x;
  int bx=blockIdx.x;
  int gIdx = bx * NPBLOCK + tx;

  // FIXME coalescence
  vel[ gIdx * 3 + coord ] += force[ data [ gIdx ] ] * dt;
}


void updatevel(coord vel[NP][NP][NP],
               float force[NP][NP][NP],
               int data[NP][NP][NP],
               int coord,
               float dt) {
  dim3 dimGriddata(NPART/NPBLOCK);
  dim3 dimBlockdata(NPBLOCK);
  P4A_launch_kernel(dimGriddata,dimBlockdata,k_updatevel,(float *)vel, (float *)force, (int *)data, coord, dt);
}

