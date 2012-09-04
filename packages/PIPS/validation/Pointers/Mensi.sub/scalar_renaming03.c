#include <stdio.h>
int scalar_renaming03()
{
  int *p, i, T, A[100], B[100], C[100], D[100];
  p = &T;
  for(i = 0; i<100; i++) {
    T = A[i] + B[i];
    C[i] = T + T;
    T = D[i] - B[i];
    *p = 1;
    A[i] = T*T;
  }

  return T;
}
