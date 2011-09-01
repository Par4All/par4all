#include <string.h>
#include <math.h>

#include "stars-pm.h"




static void _histogram(int data[NP][NP][NP],
              int histo[NP][NP][NP]) {
  int i,j,k;
#ifdef P4A_CUDA_CHEAT
  for (i = 0; i < NP; i++) {
    for (k = 0; k < NP; k++) {
      for (j = 0; j < NP; j++) {
        histo[i][j][k] = 0;
      }
    }
  }
#else
  memset(histo, 0, NPART * sizeof(int));
#endif
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

}

void histogram(int data[NP][NP][NP],
               int histo[NP][NP][NP]) {

  TIMING(_histogram(data,histo));
}

