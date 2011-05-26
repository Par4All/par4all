/* Expected result:

 */


#include <stdio.h>

int SIZE = 10;

void induction01( double A[SIZE], double B[SIZE][SIZE] )
{
  int i,j;
  int k = -1;
  int l = 10;
  for(i=0 ; i < SIZE ; i++) {
    k = k + 1;
    l+=2;
    for(j=0 ; j < SIZE ; j++) {
      A[k] = B[l][k] + A[k];
    }
  }
}


