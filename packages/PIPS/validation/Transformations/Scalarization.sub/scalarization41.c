#include <stdio.h>
#define N 10
char *name = "";
int j = 0, k = 0; // Loop indices
int a[N];
int b[N];
int tmp;


void out1() {
  b[0] = 0;
  for (j = 0; j < N; j += 1) {
    a[j] = 1;
    for (k = 0; k < N; k += 1)
      // There should be an out region out here !
      b[k] = a[j];
  }
  printf("%d", b[0]);  // use of b[0] will generated a region out before !
}

void out0() {
  b[0] = 0;
  for (j = 0; j < N; j += 1) {
    a[j] = 1;
    for (k = j; k < N; k += 1)
      // There should be an out region out here !
      b[k] = a[k];
  }
  printf("%d", b[0]);  // use of b[0] will generated a region out before !
}

