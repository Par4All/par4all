#include <math.h>

#include "stars-pm.h"

static void _updatevel(coord vel[NP][NP][NP],
                       float force[NP][NP][NP],
                       int data[NP][NP][NP],
                       int c,
                       float dt) {
#ifndef __PIPS__
  int i;
  /* Les casts ne passent pas dans PIPS :-( */
  for (i = 0; i < NPART; i++) {
    ((coord *)vel)[i]._[c] += ((float *)force)[((int*)data)[i]] * dt;
  }
#else
  int i, j, k;
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
        int x = floor(((float)data[i][j][k]) / (float)(NP * NP));
        int y = floor(((float)(data[i][j][k] - x * NP * NP))
            / (float)(NP));
        int z = data[i][j][k] - x * NP * NP - y * NP;
        vel[i][j][k]._[c] += force[x][y][z] * dt;
      }
    }
  }
#endif
}


void updatevel(coord vel[NP][NP][NP],
                     float force[NP][NP][NP],
                     int data[NP][NP][NP],
                     int c,
                     float dt) {
  TIMING(_updatevel(vel,force,data,c,dt));
}
