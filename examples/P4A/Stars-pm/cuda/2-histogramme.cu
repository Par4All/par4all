#include <string.h>
#include <math.h>

#include "stars-pm-cuda.h"


#ifndef CPUHISTO
__global__ void k_histo(int *data,
              int *histo) {
  int tx=threadIdx.x;
  int bx=blockIdx.x;

  int gIdx = bx * NPBLOCK + tx;
  atomicAdd(&histo[data[gIdx]],1);
}
#endif

void do_histo(int *data,
              int *histo) {
  int i;
  memset(histo, 0, NPART * sizeof(int));
  for (i = 0; i < NPART; i++) {
    ++histo[data[i]];
  }
}


void histogram(int data[NP][NP][NP],
               int histo[NP][NP][NP]) {
#ifndef CPUHISTO
  // Compute histogram on GPU using atomic operation
  dim3 dimGriddata(NPART/NPBLOCK);
  dim3 dimBlockdata(NPBLOCK);
  cudaMemset(histo,0,NPART*sizeof(int));
  P4A_launch_kernel(dimGriddata,dimBlockdata,k_histo,(int *)data, (int *)histo);
#else
  // Compute histogram sequentially on CPU
#ifdef P4A_TIMING
  double start_time = get_time();
#endif
  int hdata[NP][NP][NP];
  int hhisto[NP][NP][NP];
  cudaMemcpy(hdata, data, sizeof(int) * NPART, cudaMemcpyDeviceToHost);
  // Do histogramming
  memset(hhisto,0,NPART*sizeof(int));
  do_histo((int *)hdata,(int  *)hhisto);
  cudaMemcpy(histo, hhisto, sizeof(int) * NPART, cudaMemcpyHostToDevice);

#ifdef P4A_TIMING
  double end_time = get_time();
  fprintf(stderr," P4A: Time for '%s' : %fms\n",__FUNCTION__, (end_time-start_time)*1000);
#endif

#endif
}
