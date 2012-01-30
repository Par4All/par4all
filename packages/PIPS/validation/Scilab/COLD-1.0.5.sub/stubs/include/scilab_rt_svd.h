/*---------------------------------------------------- -*- C -*-
 *
 *  (c) HPC Project - 2010-2011
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
 
extern void scilab_rt_svd_d2_d2d2d2(int xm, int xn, double X[xm][xn],
	int um, int un, double U[um][un],
	int sm, int sn, double W[sm][sn],
	int vm, int vn, double V[vm][vn]);
	
extern void scilab_rt_svd_i2_d2d2d2(int xm, int xn, int X[xm][xn],
	int um, int un, double U[um][un],
	int sm, int sn, double W[sm][sn],
	int vm, int vn, double V[vm][vn]);

void scilab_rt_svd_MKL_d2_d2d2d2(int xm, int xn, double X[xm][xn],
	int um, int un, double U[um][un],
	int sm, int sn, double W[sm][sn],
	int vm, int vn, double V[vm][vn]);
	
void scilab_rt_svd_NR_d2_d2d2d2(int xm, int xn, double X[xm][xn],
	int um, int un, double U[um][un],
	int sm, int sn, double W[sm][sn],
	int vm, int vn, double V[vm][vn]);
														 
