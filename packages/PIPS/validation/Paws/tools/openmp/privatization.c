#include <stdio.h>

int privatization (void) {
  int i, j, k, l;

  for (i = 0; i < 2; ++i) {
    j = 12;
    k=22;
    for(j=4; j>0; j-=5) {
         k = 23;
         l = k*2 + j;
    }
  }

  j = l;;
}
