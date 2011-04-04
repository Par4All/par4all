/* Expected result:

 */


#include <stdio.h>

int SIZE = 10;

void induction05( double A[SIZE], double B[SIZE][SIZE] )
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

int main(int argc, char **argv)
{
  double A[SIZE], B[SIZE][SIZE];
  int i;
  for(i=0 ; i < SIZE ; i++)
    A[i] = 0.;
  induction05(A, B);
  printf ("%f\n", A[0]);
}

