// two scalarizable arrays in the same loop nest

#include <stdio.h>
#define N 10

int main() {
int j = 0, k = 0;
int a[N];
int b[N];
int c[N];
  b[0] = 0;
  for (j = 0; j < N; j += 1) {
    a[j] = 1;
    c[j] = a[j];
    for (k = 0; k < N; k += 1)
      {
	a[j] = 0;
	b[j] = a[j]+c[j];
      }
  }
  printf("%d", b[0]);
  return(0);
}

