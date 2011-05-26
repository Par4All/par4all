#include <stdlib.h>
#include <stdio.h>
#include <math.h>
/* Changed to run in less than 5 seconds on the G92 to have to finish for
   the demo: */
#define N 128
#define M 128

/* To be able to select if the computations are done in float or double: */
typedef float internal_float_t;

typedef struct	{
  internal_float_t re;
  internal_float_t im;
} complex;

complex imagein[N][N], imageout[M][M];


/*****************************************************************************/
/*****************************************************************************/

void getimage(char *filename) {
  double z,amp,sum,squares;
  int i,j,conj,nx,ny;
  unsigned char c;
  FILE *fp;

  /* There is no light on the screen at the beginning: */
  for(i=0;i<M;i++)
    for(j=0;j<M;j++) {
      imageout[i][j].re=0.0;
      imageout[i][j].im=0.0;
    }
}

int main(int argc,char *argv[]) {
  internal_float_t lambda,pixin,pixout,amp,pha,d,x,y,z,z2,z3;
  internal_float_t a,b,c,fact,pi2,twopi, centin, centout;
  int i,j,k,l;

  getimage(argv[1]);
}
