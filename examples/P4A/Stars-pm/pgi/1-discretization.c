#include "stars-pm.h"


/**
 * Compute the mapping between particles position and grid coordinates
 */
static void _discretization(coord pos[NP][NP][NP],
                    int data[NP][NP][NP]) {
  int i, j, k;
  float x, y, z;
#ifdef P4A_CUDA_CHEAT               // 0.37ms per kernel launch
  for (i = 0; i < NP; i++) {
    for (k = 0; k < NP; k++) {
      for (j = 0; j < NP; j++) {
#else                               // 7.25ms per kernel launch
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
#endif
        x = pos[i][j][k]._[0];
        y = pos[i][j][k]._[1];
        z = pos[i][j][k]._[2];
        data[i][j][k] = (int)(x / DX) * NP * NP + (int)(y / DX) * NP
            + (int)(z / DX);
      }
    }
  }

}


void discretization(coord pos[NP][NP][NP],
                    int data[NP][NP][NP]) {
  TIMING(_discretization(pos,data));
}
