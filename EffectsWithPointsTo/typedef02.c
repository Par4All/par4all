#include <stdio.h>
#include <stdlib.h>
#define SIZE 64
#define T 64

typedef float float_t1[SIZE];
typedef float float_t2[SIZE][SIZE];

float_t1 tab1[SIZE], *tab3;
float_t2 tab2;




void compute() {
  int i, j;

  for(i = 1;i < SIZE - 1; i++)
    for(j = 1;j < SIZE - 1; j++) {
      tab1[i][j] = i*j;
      tab2[i][j] = i*j;
      tab3[i][j] = i*j;
    }
  
}


int main(int argc, char *argv[]) {
  int t,i;

  for(i = 1;i < SIZE - 1; i++)
    tab3[i] = malloc(SIZE * sizeof(float));
  for(t = 0; t < T; t++)
    compute();

  for(i = 1;i < SIZE - 1; i++)
    free(tab3[i]);

  return 0;
}
