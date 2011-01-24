#include "stars-pm.h"



static void _forcex(float pot[NP][NP][NP], float fx[NP][NP][NP]) {
  int i, j, k;
#ifdef P4A_CUDA_CHEAT               // 0.17ms per kernel launch
  for (i = 0; i < NP; i++) {
    for (k = 0; k < NP; k++) {
      for (j = 0; j < NP; j++) {
#else                               // 5.78ms per kernel launch
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
#endif
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
#ifdef P4A_CUDA_CHEAT               // 0.19ms per kernel launch
  for (i = 0; i < NP; i++) {
    for (k = 0; k < NP; k++) {
      for (j = 0; j < NP; j++) {
#else                               // 5.1ms per kernel launch
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
#endif
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
#ifdef P4A_CUDA_CHEAT               // 0.19ms per kernel launch
  for (i = 0; i < NP; i++) {
    for (k = 0; k < NP; k++) {
      for (j = 0; j < NP; j++) {
#else                               // 5.5ms per kernel launch
  for (i = 0; i < NP; i++) {
    for (j = 0; j < NP; j++) {
      for (k = 0; k < NP; k++) {
#endif
        fx[i][j][k] = (pot[i][j][(k + 1) & (NP - 1)] - pot[i][j][(k - 1)
            & (NP - 1)]) / (2. * DX);
      }
    }
  }
}
void forcez(float pot[NP][NP][NP], float fx[NP][NP][NP]) {
  TIMING(_forcez(pot,fx));
}

