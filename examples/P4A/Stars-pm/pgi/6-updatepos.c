#include "stars-pm.h"

static void _updatepos(coord pos[NP][NP][NP],
               coord vel[NP][NP][NP]) {
  float posX, posY, posZ;
  int i, j, k;
#pragma acc region
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
        posX = pos[i][j][k]._[0] + vel[i][j][k]._[0] * DT;
        posY = pos[i][j][k]._[1] + vel[i][j][k]._[1] * DT;
        posZ = pos[i][j][k]._[2] + vel[i][j][k]._[2] * DT;
        pos[i][j][k]._[0] = posX + LBOX * ((posX < 0) - (posX > LBOX));
        pos[i][j][k]._[1] = posY + LBOX * ((posY < 0) - (posY > LBOX));
        pos[i][j][k]._[2] = posZ + LBOX * ((posZ < 0) - (posZ > LBOX));
      }
    }
  }

}
void updatepos(coord pos[NP][NP][NP], coord vel[NP][NP][NP]) {
  TIMING(_updatepos(pos,vel));
}

