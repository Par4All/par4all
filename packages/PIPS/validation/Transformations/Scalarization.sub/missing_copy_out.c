#include <stdio.h>


#define N 1
char *name = "";
int j = 0, k = 0; // Loop indices
int a[N];
int b[N];
int tmp;


void scalarization1() {

  b[0] = 0;
  for (j = 0; j < N; j += 1) {
    a[j] = 1;
    for (k = 0; k < N; k += 1)
      b[j] = a[j];
  }
  printf("%d", b[0]);

  /* This printf used to prevent copy out for b[] when scalarized */
  printf("%s", name);
}


void scalarization2() {
  b[0] = 0;
  for (j = 0; j < N; j += 1) {
    a[j] = 1;
    for (k = 0; k < N; k += 1)
      b[j] = a[j];
  }
  printf("%d", b[0]);
}

int main() {
  printf("Scalarization 1 : ");
  scalarization1();
  printf("\nScalarization 2 : ");
  scalarization2();
  printf("\n");
  return 0;
}
