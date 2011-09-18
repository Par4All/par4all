#include <string.h>
#include <math.h>

#include "stars-pm.h"



#pragma hmpp hmpp_histogram codelet, target=CUDA
static void hmpp_histogram(int data[NP*NP*NP],
              int histo[NP*NP*NP]) {
  int i,j,k;
  for (i = 0; i < NPART; i++) {
    histo[i] = 0;
  }
  for (i = 0; i < NPART; i++) {
    ++histo[data[i]];
  }
}

void histogram(int data[NP][NP][NP],
               int histo[NP][NP][NP]) {
#pragma hmpp hmpp_histogram callsite
  TIMING(hmpp_histogram((int *)data,(int *)histo));
}

