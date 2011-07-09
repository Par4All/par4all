/* LOCALIZE_DECLARATION should deal with structures

   Loop index should be declared outside of the loop, not inside... :-)
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define N 256
#define M 256
#define N 64
#define M 64
#define SIGIMA 1.0
#define PI M_PI
#define ZERO 1.0e-37

typedef struct	{
		float re;
		float im;
		} complex;

complex imagein[N][N], imageout[M][M];

void getimage(char *filename) {
  double z,amp,sum,squares;
  int i,j,conj,nx,ny;
  unsigned char c;
  FILE *fp;

  /* Erase the memory, in case the image is not big enough: */
  for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
      imagein[i][j].re=0.0;
      imagein[i][j].im=0.0;
    }
}
