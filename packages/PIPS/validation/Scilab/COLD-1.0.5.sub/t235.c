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

void fiff__d2(int _u_U_n0,int _u_U_n1,double _u_U[_u_U_n0][_u_U_n1])
{
  double _u_a = 2.5;
  double _u_b = 1.5;
  double _u_c = 0.5;
  int _u_n = 350;
  int _u_m = 350;
  int _tmpxx1 = (_u_n-1);
  double _u_h = (_u_a / _tmpxx1);
  int _tmpxx2 = (_u_m-1);
  double _u_k = (_u_b / _tmpxx2);
  double _tmpxx3 = (_u_c*_u_k);
  double _u_r = (_tmpxx3 / _u_h);
  double _tmpxx4 = (_u_c*_u_k);
  _u_r = (_tmpxx4 / _u_h);
  double _u_r2 = pow(_u_r,2);
  double _tmpxx5 = pow(_u_r,2);
  double _u_r22 = (_tmpxx5 / 2);
  double _tmpxx6 = pow(_u_r,2);
  double _u_s1 = (1-_tmpxx6);
  double _tmpxx7 = pow(_u_r,2);
  double _tmpxx8 = (2*_tmpxx7);
  double _u_s2 = (2-_tmpxx8);
  
  scilab_rt_zeros_i0i0_d2(_u_m,_u_n,_u_U_n0,_u_U_n1,_u_U);
  for (int _u_i1=2; _u_i1<=(_u_n-1); _u_i1++) {
    double _tmpxx9 = (scilab_rt_sin_d0_(((SCILAB_PI*_u_h)*(_u_i1-1)))+scilab_rt_sin_d0_((((2*SCILAB_PI)*_u_h)*(_u_i1-1))));
    _u_U[_u_i1-1][0] = _tmpxx9;
    double _tmpxx10 = ((_u_s1*(scilab_rt_sin_d0_(((SCILAB_PI*_u_h)*(_u_i1-1)))+scilab_rt_sin_d0_((((2*SCILAB_PI)*_u_h)*(_u_i1-1)))))+(_u_r22*(((scilab_rt_sin_d0_(((SCILAB_PI*_u_h)*_u_i1))+scilab_rt_sin_d0_((((2*SCILAB_PI)*_u_h)*_u_i1)))+scilab_rt_sin_d0_(((SCILAB_PI*_u_h)*(_u_i1-2))))+scilab_rt_sin_d0_((((2*SCILAB_PI)*_u_h)*(_u_i1-2))))));
    _u_U[_u_i1-1][1] = _tmpxx10;
  }
  for (int _u_j1=3; _u_j1<=_u_m; _u_j1++) {
    for (int _u_i1=2; _u_i1<=(_u_n-1); _u_i1++) {
      double _tmpxx11 = (((_u_s2*_u_U[_u_i1-1][(_u_j1-1)-1])+(_u_r2*(_u_U[(_u_i1-1)-1][(_u_j1-1)-1]+_u_U[(_u_i1+1)-1][(_u_j1-1)-1])))-_u_U[_u_i1-1][(_u_j1-2)-1]);
      _u_U[_u_i1-1][_u_j1-1] = _tmpxx11;
    }
  }
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t235.sce - from mcgill/fiff_function.sce */
  /* -----------------------------------------------------------------   */
  /*  - fiff.sce without function */
  /*  */
  /*  - This function finds the finite-difference solution */
  /* 	to the wave equation */
  /*  */
  /* 			     2 */
  /* 		u  (x, t) = c u  (x, t), */
  /* 		 tt	       xx */
  /*  */
  /* 	with the boundary conditions */
  /*  */
  /* 		u(0, t) = 0, u(a, t) = 0 for all 0 <= t <= b, */
  /*  */
  /* 		u(x, 0) = sin(pi*x)+sin(2*pi*x), for all 0 < x < a, */
  /*  */
  /* 		u (x, 0) = 0 for all 0 < x < a.t */
  /* ------------------------------------------------------------------- */
  scilab_rt_tic__();
  double _u_R[350][350];
  fiff__d2(350,350,_u_R);
  double _u_elapsedTime = scilab_rt_toc__();
  /* disp("Elapsed Time:"); */
  /* disp(elapsedTime); */
  double _u_acc = 0.0;
  for (int _u_ii=1; _u_ii<=350; _u_ii++) {
    for (int _u_jj=1; _u_jj<=350; _u_jj++) {
      double _tmpxx0 = _u_R[_u_ii-1][_u_jj-1];
      _u_acc = (_u_acc+_tmpxx0);
    }
  }
  scilab_rt_disp_s0_("Accumulated sum of all elements of U");
  scilab_rt_disp_d0_(_u_acc);

  scilab_rt_terminate();
}

