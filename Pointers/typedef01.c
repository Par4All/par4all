#include <stdio.h>
#include <stdlib.h>
#define SIZE 64
#define T 64

typedef float float_t;

float_t tab[SIZE][SIZE];




void compute() {
  int i, j;

  for(i = 1;i < SIZE - 1; i++)
    for(j = 1;j < SIZE - 1; j++) {
      tab[i][j] = i*j;
    }
  
}


int main(int argc, char *argv[]) {
  int t;


  for(t = 0; t < T; t++)
    compute();


  return 0;
}
