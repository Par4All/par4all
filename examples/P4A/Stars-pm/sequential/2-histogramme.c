#include <string.h>
#include <math.h>
#include <stdio.h>

#include "stars-pm.h"

void histogram(int data[NP][NP][NP],
               int histo[NP][NP][NP]) {
  int i, j, k;

#ifdef P4A_TIMING
  double end_time,start_time = get_time();
#endif


  memset(histo, 0, NPART * sizeof(int));
#ifndef P4A
  /* Les casts ne passent pas dans PIPS :-( */
  for (i = 0; i < NPART; i++) {
    ++(((int *)histo)[((int*)data)[i]]);
  }
#else
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
        int x = floor(((float)data[i][j][k]) / (float)(NP * NP));
        int y = floor(((float)(data[i][j][k] - x * NP * NP))
            / (float)(NP));
        int z = data[i][j][k] - x * NP * NP - y * NP;
        ++histo[x][y][z];
      }
    }
  }
#endif

#ifdef P4A_TIMING
  end_time = get_time();
  fprintf(stderr," P4A: Time for '%s' : %fms\n",__FUNCTION__, (end_time-start_time)*1000);
#endif


}

