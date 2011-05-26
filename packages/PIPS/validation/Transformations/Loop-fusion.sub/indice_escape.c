#include <stdio.h>

#define N 100
void loop_fusion05( int a[N], int b[N]) {
  int i,j;

  /* Loops are fusable but second indice value has to be generated */
  for( i=0; i<N; i++ ) {
    a[i] = i;
  }

  for( j=0; j<N; j++ ) {
    b[j] += a[j];
  }

  printf("Is j initialized ? %d\n",j);
}
