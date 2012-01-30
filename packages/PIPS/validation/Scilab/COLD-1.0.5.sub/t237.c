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

void mbrt__d2(int _u_SET_n0,int _u_SET_n1,double _u_SET[_u_SET_n0][_u_SET_n1])
{
  int _u_scale = 1;
  double _tmpxx1 = scilab_rt_sqrt_i0_(_u_scale);
  int _u_N = scilab_rt_round_d0_((6000*_tmpxx1));
  int _tmpxx2 = pow(10,3);
  double _tmpxx3 = scilab_rt_sqrt_i0_(_u_scale);
  int _u_Nmax = scilab_rt_round_d0_((_tmpxx2*_tmpxx3));
  double _tmpxx4 = scilab_rt_sqrt_i0_(_u_N);
  int _u_side = scilab_rt_round_d0_(_tmpxx4);
  _u_side = 77;
  double _u_ya = (-1.0);
  double _u_yb = 1.0;
  double _u_xa = (-1.5);
  double _u_xb = 0.5;
  double _tmpxx5 = (_u_xb-_u_xa);
  int _tmpxx6 = (_u_side-1);
  double _u_dx = (_tmpxx5 / _tmpxx6);
  double _tmpxx7 = (_u_yb-_u_ya);
  int _tmpxx8 = (_u_side-1);
  double _u_dy = (_tmpxx7 / _tmpxx8);
  
  scilab_rt_zeros_i0i0_d2(_u_side,_u_side,_u_SET_n0,_u_SET_n1,_u_SET);
  for (int _u_x=0; _u_x<=(_u_side-1); _u_x++) {
    for (int _u_y=0; _u_y<=(_u_side-1); _u_y++) {
      double _tmpxx9 = (_u_x*_u_dx);
      double _tmpxx10 = (_u_y*_u_dy);
      double _tmpxx11 = (_u_ya+_tmpxx10);
      double _tmpxx12 = (_u_xa+_tmpxx9);
      double complex _tmpxx13 = (I*_tmpxx11);
      double complex _u_X = (_tmpxx12+_tmpxx13);
      int _u_MAX = _u_Nmax;
      double complex _u_c = _u_X;
      int _u_i = 0;
      while ( ((scilab_rt_abs_z0_(_u_X)<2) && (_u_i<_u_MAX)) ) {
        double complex _tmpxx14 = (_u_X*_u_X);
        _u_X = (_tmpxx14+_u_c);
        _u_i = (_u_i+1);
      }
      _u_SET[(_u_y+1)-1][(_u_x+1)-1] = _u_i;
    }
  }
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t237.sce - from mcgill/mbrt_function.sce */
  scilab_rt_tic__();
  double _u_SET[77][77];
  mbrt__d2(77,77,_u_SET);
  double _u_elapsedTime = scilab_rt_toc__();
  /* disp("Elapsed Time: "); */
  /* disp(elapsedTime); */
  double _u_acc = 0;
  for (int _u_i=1; _u_i<=77; _u_i++) {
    for (int _u_j=1; _u_j<=77; _u_j++) {
      double _tmpxx0 = _u_SET[_u_i-1][_u_j-1];
      _u_acc = (_u_acc+_tmpxx0);
    }
  }
  scilab_rt_disp_s0_("Accumulated sum of all elements of the array");
  scilab_rt_disp_d0_(_u_acc);

  scilab_rt_terminate();
}

