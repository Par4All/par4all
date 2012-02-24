
#include <math.h>

void lyapunov_init (int xDim, int yDim, double *u3, double *v3, double dt_in, double scale_in, double *bbox);
void lyapunov_iterate (int xDim, int yDim, double *u3, double *v3);
void lyapunov_finish (int np1, int np2, double *ly2);

void lyapunov_init_c99 (int xDim, int yDim, int np1, int np2,
			double u1[xDim][yDim], double v1[xDim][yDim],
			double u2[xDim][yDim], double v2[xDim][yDim],
			double u3[xDim][yDim], double v3[xDim][yDim],
			double xp[np1][np2], double yp[np1][np2],
			double dt, int scale, double bbox_in[4]);
			
void lyapunov_iterate_c99 (int xDim, int yDim, int np1, int np2,
			double u1[xDim][yDim], double v1[xDim][yDim],
			double u2[xDim][yDim], double v2[xDim][yDim],
			double u3[xDim][yDim], double v3[xDim][yDim],
			double xp[np1][np2], double yp[np1][np2],
			double dt, int scale, double bbox_in[4]);
			
void lyapunov_finish_c99 (int np1, int np2,
			double xp[np1][np2], double yp[np1][np2],
			double dt, int scale, double bbox_in[4],
			double ly2[np1][np2]);

double vinterp (int xDim, int yDim, double *w, double x, double y);
