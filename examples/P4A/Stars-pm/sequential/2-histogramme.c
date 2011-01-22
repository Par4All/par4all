#include <string.h>
#include <math.h>

#include "stars-pm.h"

void histogram(int data[NP][NP][NP],
               int histo[NP][NP][NP]) {
  int i, j, k;
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
}

