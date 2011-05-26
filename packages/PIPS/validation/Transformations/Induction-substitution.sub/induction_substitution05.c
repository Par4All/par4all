/* Expected result:

 */


#include <stdio.h>


void induction05(int SIZE, double A[SIZE], double B[SIZE][SIZE] )
{
  int i,j;
  int k = SIZE;
  for(i=0 ; i < SIZE ; i++) {
    if(k--) {
      A[k] = B[j-k][k] + A[k];
    }
    if(--k) {
      A[k] = B[j-k][k] + A[k];
    }
  }
}


