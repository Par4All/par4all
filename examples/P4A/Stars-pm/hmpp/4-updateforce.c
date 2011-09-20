#include "stars-pm.h"



#pragma hmpp _forcex codelet, target=CUDA
static void _forcex(float pot[NP][NP][NP], float fx[NP][NP][NP]) {
  int i, j, k;
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
        fx[i][j][k] = (pot[(i + 1) & (NP - 1)][j][k] - pot[(i - 1) & (NP
            - 1)][j][k]) / (2. * DX);
      }
    }
  }
}

void forcex(float pot[NP][NP][NP], float fx[NP][NP][NP]) {
#pragma hmpp _forcex callsite
  TIMING(_forcex(pot,fx));
}


//========================================================================
#pragma hmpp _forcey codelet, target=CUDA
static void _forcey(float pot[NP][NP][NP], float fx[NP][NP][NP]) {
  int i, j, k;
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
        fx[i][j][k] = (pot[i][(j + 1) & (NP - 1)][k] - pot[i][(j - 1)
            & (NP - 1)][k]) / (2. * DX);
      }
    }
  }
}
void forcey(float pot[NP][NP][NP], float fx[NP][NP][NP]) {
#pragma hmpp _forcey callsite
  TIMING(_forcey(pot,fx));
}

//========================================================================
#pragma hmpp _forcez codelet, target=CUDA
static void _forcez(float pot[NP][NP][NP], float fx[NP][NP][NP]) {
  int i, j, k;
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
        fx[i][j][k] = (pot[i][j][(k + 1) & (NP - 1)] - pot[i][j][(k - 1)
            & (NP - 1)]) / (2. * DX);
      }
    }
  }
}
void forcez(float pot[NP][NP][NP], float fx[NP][NP][NP]) {
#pragma hmpp _forcez callsite
  TIMING(_forcez(pot,fx));
}

