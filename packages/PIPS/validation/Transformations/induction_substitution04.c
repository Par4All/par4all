/* Expected result:

 */


#include <stdio.h>

int SIZE = 10;

void induction04( double A[SIZE], double B[SIZE][SIZE] )
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

int main(int argc, char **argv)
{
  double A[SIZE], B[SIZE][SIZE];
  int i;
  for(i=0 ; i < SIZE ; i++)
    A[i] = 0.;
  induction04(A, B);
  printf ("%f\n", A[0]);
}

