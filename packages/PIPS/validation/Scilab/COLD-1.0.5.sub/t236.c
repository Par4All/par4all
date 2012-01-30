/*
 * (c) HPC Project - 2010-2011 - All rights reserved
 *
 */

#include "scilab_rt.h"


int __lv0;
int __lv1;
int __lv2;
int __lv3;

/*----------------------------------------------------*/


/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t236.sce - from mcgill/mbrt.sce */
  /* computes mandelbrot set */
  /* tic(); */
  int _u_scale = 1;
  double _tmpxx0 = scilab_rt_sqrt_i0_(_u_scale);
  int _u_N = scilab_rt_round_d0_((6000*_tmpxx0));
  int _tmpxx1 = pow(10,3);
  double _tmpxx2 = scilab_rt_sqrt_i0_(_u_scale);
  int _u_Nmax = scilab_rt_round_d0_((_tmpxx1*_tmpxx2));
  double _tmpxx3 = scilab_rt_sqrt_i0_(_u_N);
  int _u_side = scilab_rt_round_d0_(_tmpxx3);
  _u_side = 77;
  double _u_ya = (-1.0);
  double _u_yb = 1.0;
  double _u_xa = (-1.5);
  double _u_xb = 0.5;
  double _tmpxx4 = (_u_xb-_u_xa);
  int _tmpxx5 = (_u_side-1);
  double _u_dx = (_tmpxx4 / _tmpxx5);
  double _tmpxx6 = (_u_yb-_u_ya);
  int _tmpxx7 = (_u_side-1);
  double _u_dy = (_tmpxx6 / _tmpxx7);
  double _u_SET[77][77];
  scilab_rt_zeros_i0i0_d2(_u_side,_u_side,77,77,_u_SET);
  for (int _u_x=0; _u_x<=(_u_side-1); _u_x++) {
    for (int _u_y=0; _u_y<=(_u_side-1); _u_y++) {
      double _tmpxx8 = (_u_x*_u_dx);
      double _tmpxx9 = (_u_y*_u_dy);
      double _tmpxx10 = (_u_ya+_tmpxx9);
      double _tmpxx11 = (_u_xa+_tmpxx8);
      double complex _tmpxx12 = (I*_tmpxx10);
      double complex _u_X = (_tmpxx11+_tmpxx12);
      int _u_MAX = _u_Nmax;
      double complex _u_c = _u_X;
      int _u_i = 0;
      while ( ((scilab_rt_abs_z0_(_u_X)<2) && (_u_i<_u_MAX)) ) {
        double complex _tmpxx13 = (_u_X*_u_X);
        _u_X = (_tmpxx13+_u_c);
        _u_i = (_u_i+1);
      }
      _u_SET[(_u_y+1)-1][(_u_x+1)-1] = _u_i;
    }
  }
  /* elapsedTime = toc(); */
  /* disp("Elapsed Time: "); */
  /* disp(elapsedTime); */
  double _u_acc = 0;
  for (int _u_i=1; _u_i<=77; _u_i++) {
    for (int _u_j=1; _u_j<=77; _u_j++) {
      double _tmpxx14 = _u_SET[_u_i-1][_u_j-1];
      _u_acc = (_u_acc+_tmpxx14);
    }
  }
  scilab_rt_disp_s0_("Accumulated sum of all elements of the array: ");
  scilab_rt_disp_d0_(_u_acc);

  scilab_rt_terminate();
}

