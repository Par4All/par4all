#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "lyapunov.h"
#define DEFAULT -9999.

// pour debug
//#include <sys/types.h>
//#include <signal.h>

/* contants */
const double earth_radius=6378137.; /*earth radius */
const double NaN = 0.0 / 0.0;


double t=0.0;
double dxl, dt;
// bbox
double lat1,lon1,lat2,lon2;


// #define M_PI	3.1416

#ifdef MATLAB
#define printf mexPrintf
#endif

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif


/* 
* INPUT :
* xDim : number of columns
* yDim : number of lines
* u3 (xDim,yDim, tDim) : zonal component of current
* v3 (xDim, yDim) : meridian component of current
*/

//void lyapunov_init_c99 (int xDim, int yDim, double *u3_1d, double *v3_1d, double dt_in, double scale_in, double *bbox_in)
void lyapunov_init_c99 (int xDim, int yDim, int np1, int np2,
			double u1[xDim][yDim], double v1[xDim][yDim],
			double u2[xDim][yDim], double v2[xDim][yDim],
			double u3[xDim][yDim], double v3[xDim][yDim],
			double xp[np1][np2], double yp[np1][np2],
			double dt, int scale, double bbox_in[4])
{
	
	int i, j, p, q;

	//int n1 = xDim;
	//int n2 = yDim;
	
	printf("(C) dt=%f scale=%d\n", dt, scale);
	printf("(C) bbox = [ %f %f %f %f ]\n", bbox_in[0], bbox_in[1], bbox_in[2], bbox_in[3]);
	
	lat1=bbox_in[0];
	lon1=bbox_in[1];
	lat2=bbox_in[2];
	lon2=bbox_in[3];	
	
	// cast to c99 VLA
// 	double u1[n1][n2]=(double (*)[n1][n2])u1_1d;
// 	double v1[n1][n2]=(double (*)[n1][n2])v1_1d;
// 	double u2[n1][n2]=(double (*)[n1][n2])u2_1d;
// 	double v2[n1][n2]=(double (*)[n1][n2])v2_1d;
// 	double u3[n1][n2]=(double (*)[n1][n2])u3_1d;
// 	double v3[n1][n2]=(double (*)[n1][n2])v3_1d;
// 	double xp[np1][np2]=(double (*)[np1][np2])xp_1d;
// 	double yp[np1][np2]=(double (*)[np1][np2])yp_1d;
	
	
//	dt = 43200.0;	// pourquoi 12h alors que dt est 24h ? TODO: parameter
	
	dxl = 0.5;  // TODO: compute from bbox
	
	for (p = 0; p < np1; p++) 
	{
		for (q = 0; q < np2; q++)
		{
			xp[p][q]=p * dxl / (double)scale - 180. ;
			yp[p][q]=q * dxl / (double)scale - 90. ;
		}
	}
	
	
	for (i = 0; i < xDim; i++)
	{
		for (j = 0; j < yDim; j++)
		{
			// remove land cells
			if (u3[i][j] != DEFAULT)
			{
				u1[i][j]=u3[i][j];
				v1[i][j]=v3[i][j];
			}
			else
			{
				u1[i][j]=0;
				v1[i][j]=0;
				u2[i][j]=0;
				v2[i][j]=0;
			}
		}
	}
}



//void lyapunov_iterate_c99 (int xDim, int yDim, double u3[xDim][yDim], double v3[xDim][yDim])
void lyapunov_iterate_c99 (int xDim, int yDim, int np1, int np2,
			double u1[xDim][yDim], double v1[xDim][yDim],
			double u2[xDim][yDim], double v2[xDim][yDim],
			double u3[xDim][yDim], double v3[xDim][yDim],
			double xp[np1][np2], double yp[np1][np2],
			double dt, int scale, double bbox_in[4])
{
	
	double  rat;
	double a1, b1, a2, b2, a3, b3, a4, b4;
	int i, j;
	int p,q;
	double lon, lat;
	// int n1,n2;	
	// n1 = xDim;
	// n2 = yDim;
	
	printf("(C) iterate1 modif2 xDim=%d yDim=%d\n",xDim,yDim);
	for (i = 0; i < xDim; i++)   
	{
		for (j = 0; j < yDim; j++)  
		{
			//printf("i=%d j=%d\n",i,j);
			u2[i][j]=(u1[i][j]+u3[i][j])/2.0;
			v2[i][j]=(v1[i][j]+v3[i][j])/2.0;
		}
	}
	
	rat = M_PI / 180.0;
	
	printf("(C) iterate2\n");
	for (p = 0; p < np1; p++)
	{
		for (q = 0; q < np2; q++)
		{
			lon = xp[p][q];
			lat = yp[p][q];
			
			// compute lyapunov even if hight latitude. we must use bbox to not compute it. 
			// if( (lat>= -80.0) && (lat <= 80.0))
			// {
				
				a1 = dt * vinterp (xDim, yDim, (double *)u1, lon, lat) / earth_radius / rat / cos (lat * rat);
				
				b1 = dt * vinterp (xDim, yDim, (double *)v1, lon, lat) / earth_radius / rat;
				
				a2 = dt * vinterp (xDim, yDim, (double *)u2, lon + a1 / 2.0, lat + b1 / 2.0) / earth_radius / rat / cos ((lat + b1 / 2.0) * rat);
					      
					      
				b2 = dt * vinterp (xDim, yDim, (double *)v2, lon + a1 / 2.0, lat + b1 / 2.0) / earth_radius / rat;
					      
				a3 = dt * vinterp (xDim, yDim, (double *)u2, lon + a2 / 2.0, lat + b2 / 2.0) / earth_radius / rat / cos ((lat + b2 / 2.0) * rat);
					      
					      
				b3 = dt * vinterp (xDim, yDim, (double *)v2, lon + a2 / 2.0, lat + b2 / 2.0) / earth_radius / rat;
					      
					      
				a4 = dt * vinterp (xDim, yDim, (double *)u3, lon + a3, lat + b3) / earth_radius / rat / cos ((lat + b3) * rat);
							    
				b4 = dt * vinterp (xDim, yDim, (double *)v3, lon + a3, lat + b3) / earth_radius / rat;
							    
				xp[p][q] = lon + (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0;
				yp[p][q] = lat + (b1 + 2.0 * b2 + 2.0 * b3 + b4) / 6.0;
							    
							    
			// } // if lat
		}  // for p
	}  // for q
	
	printf("(C) iterate3\n");
	t += dt;
	
	for (i = 0; i < xDim; i++)
	{
		for (j = 0; j < yDim; j++)
		{
			u1[i][j] = u3[i][j];
			v1[i][j] = v3[i][j];
			
		}
	}
}



void lyapunov_finish_c99 (int np1, int np2,
			double xp[np1][np2], double yp[np1][np2],
			double dt, int scale, double bbox_in[4],
			double ly2[np1][np2]) 
{
	double a, b, c, d, e, f, g, h;
	int i,j;
	int il,ir; // left & right indices
	int jl,jr;
	
	
	// init to DEFAULT (inutile...)
	//for (i = 0; i < np1; i++)
	//{
	//	for (j = 0; j < np2; j++)
	//	{
	//		(*ly2)[i][j]=DEFAULT;
	//	}
	//}
	
	
	for (i = 0; i < np1; i++)
	{
		for (j = 0; j < np2; j++)
		{
			// cycle on longitude axis ( %)
			ir=(i+1) % np1;
			il=(i-1+np1) % np1;   // want it > 0
			a = 2 * (xp[ ir ][ j ] - xp[ il ][ j ]) / dxl;
			c = 2 * (yp[ ir ][ j ] - yp[ il ][ j ]) / dxl;
			
			// no cycle in latitude axis ( min, max)
			jr=min( j+1, np2 );
			jl=max ( j-1, 0 );
			b = 2 * (xp[ i ][ jr ] - xp[ i ][ jl ]) / dxl;
			d = 2 * (yp[ i ][ jr ] - yp[ i ][ jl ]) / dxl;
			
			
			e = a * a + c * c;
			f = a * b + c * d;
			g = b * b + d * d;
			
			h = ((e + g) + sqrt ((e + g) * (e + g) - 4.0 * (-f * f + e * g))) / 2.0;
			
			ly2[i][j]=log (sqrt (h)) / t;
		} // for j
	} // for i
	
}


/* fonctions devant etre appelées a partir d'un kernel cuda */

double vinterp (int n1, int n2, double *w, double x, double y)
{
	
	//if(isnan(x)) x=y;
	//if(isnan(y)) y=x;
	//if( isnan(x) || isnan(y)) return NaN;
	
	int i, j, i1, i2, j1, j2;
	double cy1, cy2, c1, c2;
	double dx, dy;
	double x1, x2, y1, y2;
	double w1, w2;
	double wint;
	
	dx = 0.5;
	dy = 0.5;
	
	// take care of date change line
	if (x >= 180.0)
	{
		x = x - 360;
	}
	if (x <= -180.0)
	{
		x = x + 360;
	}
	
	
	
	i = (int) ((x + 180.0) / dx);
	j = (int) ((y + 90.0) / dy);
	
	if( (i<0) || (i>n1) || (j<0) || (j>n2) ) {
		// out of bound
		return -9999.;
	}
	
	
	i1 = i;
	i2 = i + 1;
	if (i2 == n1)
		i2 = 0;
	
	j1 = j;
	j2 = j + 1;
	if (j2 == n2)
		j2 = n2 - 1;
	
	//if (w[j2+n2*i2] == 0 || w[j1+n2*i2]  == 0 || w[j2+n2*i1] == 0 || w[j1+n2*i1] == 0)
	//return 0;
	
	x1 = i * dx - 180.0;
	x2 = (i + 1) * dx - 180.0;
	y1 = j * dx - 90.0;
	y2 = (j + 1) * dx - 90.0;
	
	cy1 = (y - y1) / (y2 - y1);
	cy2 = (y - y2) / (y1 - y2);
	
	//w1 = cy1 * w[i2][j2] + cy2 * w[i2][j1];
	//w2 = cy1 * w[i1][j2]  + cy2 * w[i1][j1];
	w1=cy1*w[j2+n2*i2]+cy2*w[j1+n2*i2];
	w2=cy1*w[j2+n2*i1]+cy2*w[j1+n2*i1];

	
	c1 = (x - x1) / (x2 - x1);
	
	c2 = (x - x2) / (x1 - x2);
	
	
	
	wint = c1 * w1 + c2 * w2;
	
	return wint;
}
