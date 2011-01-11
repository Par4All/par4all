#include <string.h>
#include <math.h>

#include "stars-pm.h"

void histogram(int data[NP][NP][NP],
               int histo[NP][NP][NP]) {

  int i;
  memset(histo, 0, NPART * sizeof(int));
  for (i = 0; i < NPART; i++) {
    ++(((int *)histo)[((int*)data)[i]]);
  }

}


