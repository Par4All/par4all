/* Expected result:

 */


#include <stdio.h>


void induction02( int SIZE, double A[SIZE], double B[SIZE][SIZE] )
{
  int i,j;
  int k = -1;
  for(i=0 ; i < SIZE ; i++) {
    k = i;
    for(j=0 ; j < SIZE ; j++) {
      k++;
      A[k] = B[j-k][k] + A[k];
    }
  }
}

