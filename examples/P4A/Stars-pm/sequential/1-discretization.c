#include "stars-pm.h"


/**
 * Compute the mapping between particles position and grid coordinates
 */
void discretization(coord pos[NP][NP][NP],
                    int data[NP][NP][NP]) {
  int i, j, k;
  float x, y, z;
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
        x = pos[i][j][k]._[0];
        y = pos[i][j][k]._[1];
        z = pos[i][j][k]._[2];
        data[i][j][k] = (int)(x / DX) * NP * NP + (int)(y / DX) * NP
            + (int)(z / DX);
      }
    }
  }

}

