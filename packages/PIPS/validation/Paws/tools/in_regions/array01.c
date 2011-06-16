#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define N 64
#define M 64
#define SIGIMA 1.0
#define PI M_PI
#define ZERO 1.0e-37

typedef struct	{
		float re;
		float im;
		} complex;

complex image[N][N];


void compute(complex image[N][N])
{
  int i,j;

  for(i=0;i<N;i++) 	
    for(j=0;j<N;j++) 
      {
	image[i][j].re = (i+j) * PI;
	image[i][j].im = ZERO;
      }
}

int main() 
{

  int i,j;

  compute(image);

  for(i=0;i<N;i++) 	
    for(j=0;j<N;j++) 
      {
	fprintf(stdout, "[%d][%d].re = %f\n", i,j, image[i][j].re);
      }

  return 0;

}
