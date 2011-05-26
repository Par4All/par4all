/* This exemple used to end with a pips error when computing 
   TOTAL_PRECONDITIONS_INTRA */

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

int main(int argc, char **argv)
{
  double A[SIZE], B[SIZE][SIZE];
  int i;
  for(i=0 ; i < SIZE ; i++)
    A[i] = 0.;
  induction01(A, B);
  printf ("%f\n", A[0]);
}

