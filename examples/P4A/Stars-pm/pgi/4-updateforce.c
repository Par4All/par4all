#include "stars-pm.h"



static void _forcex(float pot[NP][NP][NP], float fx[NP][NP][NP]) {
  int i, j, k;
#pragma acc region
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
  TIMING(_forcex(pot,fx));
}


//========================================================================
static void _forcey(float pot[NP][NP][NP], float fx[NP][NP][NP]) {
  int i, j, k;
#pragma acc region
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
  TIMING(_forcey(pot,fx));
}

//========================================================================
static void _forcez(float pot[NP][NP][NP], float fx[NP][NP][NP]) {
  int i, j, k;
#pragma acc region
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
  TIMING(_forcez(pot,fx));
}

