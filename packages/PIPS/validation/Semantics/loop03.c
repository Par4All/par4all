#include <stdlib.h>
#include <stdio.h>
/* To have M_PI: */
#define __USE_XOPEN
#include <math.h>
#define N 64
#define M 256
#define SIGIMA 1.0
#define PI M_PI
#define ZERO 1.0e-37

typedef float f_float_t;

typedef struct	{
		float re;
		float im;
		} complex;

float imagein_re[N][N], imageout_re[M][M];
float imagein_im[N][N], imageout_im[M][M];

void getimage(filename)

char *filename;

{	extern complex imagein[N][N], imageout[M][M];
	double z,amp,sum,squares;
	int i,j,conj,nx,ny;
	unsigned char c;
	FILE *fp;

	for(i=0;i<N;i++)
	{	for(j=0;j<N;j++)
		{       imagein_re[i][j]=0.0;
			imagein_im[i][j]=0.0;
		}
	}

	for(i=0;i<M;i++)
	{	for(j=0;j<M;j++)
		{       imageout_re[i][j]=0.0;
			imageout_im[i][j]=0.0;
		}
	}

	fp=fopen(filename,"r");
	fclose(fp);
}
