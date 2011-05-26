/* Expected result:

 */


#include <stdio.h>


void induction04( int SIZE, double A[SIZE], double B[SIZE][SIZE] )
{
  int i,j;
  int k = -1;
  int sum;
  for(i=0 ; i < SIZE ; i++) {
    k = i;
    for(j=0 ; j < SIZE ; j++) {
      sum = B[j-k][k] + A[k];
      A[k++] = sum;
    }
  }
}

