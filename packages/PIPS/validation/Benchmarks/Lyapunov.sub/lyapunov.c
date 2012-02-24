/* lyapunov_c99_wrapper */
/* le mex est en vieux C ansi, et lyapunov.c en c99 */

#include<stdlib.h>
#include "lyapunov.h"



// linear array (will be casted to c99 declaration)
int *ind1, *ind2;  
double *xp_1d, *yp_1d ,*xp1_1d, *yp1_1d;  
double *u1_1d, *v1_1d, *u2_1d, *v2_1d;

int scale;  // scale factor of result matrix 
double dt;
double (*bbox)[4];

void lyapunov_init (int xDim, int yDim, double *u3, double *v3, double dt_in, double scale_in, double *bbox_in)
{
	int n1 = xDim;
	int n2 = yDim;
	
	scale=(int)scale_in;
	dt=dt_in;
	bbox=(double (*)[4])bbox_in;
	
	int np1 = scale * (n1 - 1) + 1;
	int np2 = scale * (n2 - 1) + 1;
	
	
	// global tab allocation (linear)
	u1_1d = malloc (n1 * n2 * sizeof (double));
	u2_1d = malloc (n1 * n2 * sizeof (double));
	v1_1d = malloc (n1 * n2 * sizeof (double));
	v2_1d = malloc (n1 * n2 * sizeof (double));	
	xp_1d = malloc (np1 * np2 * sizeof (double));
	yp_1d = malloc (np1 * np2 * sizeof (double));
	
	
	
	lyapunov_init_c99( xDim, yDim, np1, np2, 
			   *(double(*)[xDim][yDim])u1_1d, *(double(*)[xDim][yDim])v1_1d, 
			   *(double(*)[xDim][yDim])u2_1d, *(double(*)[xDim][yDim])v2_1d,  
			   *(double(*)[xDim][yDim])u3, *(double(*)[xDim][yDim])v3,
			   *(double(*)[np1][np2])xp_1d, *(double(*)[np1][np2])yp_1d,
			   dt, scale, *bbox  );

}


void
lyapunov_iterate (int xDim, int yDim, double *u3, double *v3)
{
	//lyapunov_iterate_c99( xDim, yDim, *(double(*)[xDim][yDim])u3, *(double(*)[xDim][yDim])v3 );
	int n1 = xDim;
	int n2 = yDim;
	
	int np1 = scale * (n1 - 1) + 1;
	int np2 = scale * (n2 - 1) + 1;
	lyapunov_iterate_c99( xDim, yDim, np1, np2, 
			   *(double(*)[xDim][yDim])u1_1d, *(double(*)[xDim][yDim])v1_1d, 
			   *(double(*)[xDim][yDim])u2_1d, *(double(*)[xDim][yDim])v2_1d,  
			   *(double(*)[xDim][yDim])u3, *(double(*)[xDim][yDim])v3,
			   *(double(*)[np1][np2])xp_1d, *(double(*)[np1][np2])yp_1d,
			   dt, scale, *bbox);
}

void lyapunov_finish (int np1, int np2, double *ly2) {

	//lyapunov_finish_c99 (np1,np2,*(double(*)[np1][np2])ly2);
	lyapunov_finish_c99 (np1, np2, 
			   *(double(*)[np1][np2])xp_1d, *(double(*)[np1][np2])yp_1d,
			   dt, scale, *bbox, *(double(*)[np1][np2])ly2);
	
	free (u1_1d);
	free (v1_1d);
	free (u2_1d);
	free (v2_1d);
	
	free (xp_1d);
	free (yp_1d);
	free (xp1_1d);
	free (yp1_1d);
}
