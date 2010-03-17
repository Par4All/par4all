#include <stdlib.h>

const int N=10;
const int Z=10;

int **Matrix;
//char* filename;

int main (void) {
  int i,j;

  Matrix = (int **)malloc(N*sizeof(int *));
  for ( i = 0; i< N; i++)
    {
      Matrix[i] = (int *)malloc(Z*sizeof(int));
      for ( j = 0; j < Z; j++)
	{
	  Matrix[i][j] = i * j;
	}
    }
  return 0;
}

