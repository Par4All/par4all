#include "stars-pm-cuda.h"
#include "kernel_tools.h"


__global__ void k_discretization(float *pos, int *data) {
  int tx=threadIdx.x;
  int bx=blockIdx.x;

  __shared__ float spot[3*NPBLOCK];

  // Charge les positions depuis la mémoire globale vers la mémoire partagée
  int dataPos = bx*NPBLOCK + tx; // position du thread dans data
  int basePos = dataPos * 3; // position du thread dans pos

  spot[tx] = pos[ basePos + tx ]; // FIXME check alignement et conflit de banc
  spot[tx+NPBLOCK] = pos[ basePos + NPBLOCK + tx ];
  spot[tx+2*NPBLOCK] = pos[ basePos + NPBLOCK * 2 + tx ];

  __syncthreads();

  // data[particules] = numéro de la cellule
  data[dataPos]=(int)( spot[tx] / DX ) * NP * NP + (int)(spot[tx+1] / DX ) * NP + (int) spot[tx+2] / DX;
}


/**
 * Compute the mapping between particles position and grid coordinates
 */
void discretization(coord pos[NP][NP][NP],
                    int data[NP][NP][NP]) {

  dim3 dimGriddata(NPART/NPBLOCK);
  dim3 dimBlockdata(NPBLOCK);
  P4A_launch_kernel(dimGriddata,dimBlockdata,k_discretization,(float *)pos,(int *)data);
}

