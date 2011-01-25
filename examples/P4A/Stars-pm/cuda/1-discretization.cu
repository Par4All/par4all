#include "stars-pm-cuda.h"
#include "kernel_tools.h"


__global__ void k_discretization(float *pos, int *data) {
  int tx=threadIdx.x;
  int bx=blockIdx.x;

  __shared__ float spot[3*NPBLOCK];

  // Charge les positions depuis la mémoire globale vers la mémoire partagée
  int dataPos = bx*NPBLOCK; // position du block dans data
  int basePos = dataPos * 3; // position du block dans pos

  spot[tx]           = pos[ basePos + tx ];
  spot[tx+NPBLOCK]   = pos[ basePos + tx + NPBLOCK ];
  spot[tx+2*NPBLOCK] = pos[ basePos + tx + NPBLOCK * 2 ];

  __syncthreads();

  // data[particules] = numéro de la cellule
  data[dataPos+tx]=(int)( spot[tx*3] / DX ) * NP * NP + (int)(spot[tx*3+1] / DX ) * NP + (int)( spot[tx*3+2] / DX);

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

