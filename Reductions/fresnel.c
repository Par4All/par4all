#include <stdio.h>

enum { M = 256 };
#define M 256

void output()
{	double z,max,sum,squares;
	int i,j;
	float imageout_re[M][M];
	max=0.0;sum=0.0;squares=0.0;
	for(i=0;i<M;i++)
	{	for(j=0;j<M;j++)
		{	z=imageout_re[i][j];
		        sum+=z;
			squares+=(z*z);
			if(z>max)
			  max=z;
		}
	}

	printf("Sum after=%f  Sumsquares after=%f\n",sum,squares);
}
