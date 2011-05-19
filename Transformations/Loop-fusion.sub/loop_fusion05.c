#include <stdio.h>

#define N 100
void loop_fusion05( int a[N], int b[N]) {
  int i,j;

  /* The first loop can't be fused */
  for( i=0; i<N; i++ ) {
    a[i] = i;
  }

  for( j=0; j<N; j++ ) {
    b[j] += a[j+1];
  }

  printf("Is j initialized ? %d\n",j);
}
