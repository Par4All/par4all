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

  /*  t226.sce - from mcgill/dich.sce */
  /* This function finds the Dirichlet solution to */
  /* Laplace's equation */
  /* tic(); */
  double _u_a = 4.0;
  double _u_b = 4.0;
  double _u_h = 0.03;
  int _tmpxx0 = (-5);
  int _u_tol = pow(10,_tmpxx0);
  int _u_max1 = 1000;
  int _u_f1 = 20;
  int _u_f2 = 180;
  int _u_f3 = 80;
  int _u_f4 = 0;
  int _tmpxx1 = scilab_rt_fix_d0_((_u_a / _u_h));
  int _u_n = (_tmpxx1+1);
  int _tmpxx2 = scilab_rt_fix_d0_((_u_b / _u_h));
  int _u_m = (_tmpxx2+1);
  _u_n = 134;
  _u_m = 134;
  int _tmpxx3 = (_u_f1+_u_f2);
  int _tmpxx4 = (_u_f3+_u_f4);
  double _tmpxx5 = (_u_a*_tmpxx3);
  double _tmpxx6 = (_u_b*_tmpxx4);
  double _tmpxx7 = (2*_u_a);
  double _tmpxx8 = (2*_u_b);
  double _tmpxx9 = (_tmpxx5+_tmpxx6);
  double _tmpxx10 = (_tmpxx7+_tmpxx8);
  double _u_ave = (_tmpxx9 / _tmpxx10);
  double _tmpxx11[134][134];
  scilab_rt_ones_i0i0_d2(_u_n,_u_m,134,134,_tmpxx11);
  double _u_U[134][134];
  scilab_rt_mul_d0d2_d2(_u_ave,134,134,_tmpxx11,134,134,_u_U);
  for (int _u_l=1; _u_l<=_u_m; _u_l++) {
    _u_U[0][_u_l-1] = _u_f3;
    _u_U[_u_n-1][_u_l-1] = _u_f4;
  }
  for (int _u_k=1; _u_k<=_u_n; _u_k++) {
    _u_U[_u_k-1][0] = _u_f1;
    _u_U[_u_k-1][_u_m-1] = _u_f2;
  }
  double _tmpxx12 = ((_u_U[0][1]+_u_U[1][0]) / 2);
  _u_U[0][0] = _tmpxx12;
  double _tmpxx13 = ((_u_U[0][(_u_m-1)-1]+_u_U[1][_u_m-1]) / 2);
  _u_U[0][_u_m-1] = _tmpxx13;
  double _tmpxx14 = ((_u_U[(_u_n-1)-1][0]+_u_U[_u_n-1][1]) / 2);
  _u_U[_u_n-1][0] = _tmpxx14;
  double _tmpxx15 = ((_u_U[(_u_n-1)-1][_u_m-1]+_u_U[_u_n-1][(_u_m-1)-1]) / 2);
  _u_U[_u_n-1][_u_m-1] = _tmpxx15;
  int _tmpxx16 = (_u_n-1);
  int _tmpxx17 = (_u_m-1);
  double _tmpxx18 = scilab_rt_cos_d0_((SCILAB_PI / _tmpxx16));
  double _tmpxx19 = scilab_rt_cos_d0_((SCILAB_PI / _tmpxx17));
  double _tmpxx20 = (_tmpxx18+_tmpxx19);
  double _tmpxx21 = pow(_tmpxx20,2);
  double _tmpxx22 = scilab_rt_sqrt_d0_((4-_tmpxx21));
  double _tmpxx23 = (2+_tmpxx22);
  double _u_w = (4 / _tmpxx23);
  double _u_err = 1;
  int _u_cnt = 0;
  while ( ((_u_err>_u_tol) && (_u_cnt<=_u_max1)) ) {
    _u_err = 0;
    for (int _u_l=2; _u_l<=(_u_m-1); _u_l++) {
      for (int _u_k=2; _u_k<=(_u_n-1); _u_k++) {
        double _tmpxx24 = _u_U[_u_k-1][(_u_l+1)-1];
        double _tmpxx25 = _u_U[_u_k-1][(_u_l-1)-1];
        double _tmpxx26 = (_tmpxx24+_tmpxx25);
        double _tmpxx27 = _u_U[(_u_k+1)-1][_u_l-1];
        double _tmpxx28 = (_tmpxx26+_tmpxx27);
        double _tmpxx29 = _u_U[(_u_k-1)-1][_u_l-1];
        double _tmpxx30 = _u_U[_u_k-1][_u_l-1];
        double _tmpxx31 = (_tmpxx28+_tmpxx29);
        double _tmpxx32 = (4*_tmpxx30);
        double _tmpxx33 = (_tmpxx31-_tmpxx32);
        double _tmpxx34 = (_u_w*_tmpxx33);
        double _u_relx = (_tmpxx34 / 4);
        double _tmpxx35 = (_u_U[_u_k-1][_u_l-1]+_u_relx);
        _u_U[_u_k-1][_u_l-1] = _tmpxx35;
        double _tmpxx36 = scilab_rt_abs_d0_(_u_relx);
        if ((_u_err<=_tmpxx36)) {
          _u_err = scilab_rt_abs_d0_(_u_relx);
        }
      }
    }
    _u_cnt = (_u_cnt+1);
  }
  /* elapsedTime = toc(); */
  /* disp("Elapsed Time: "); */
  /* disp(elapsedTime); */
  double _u_acc = 0;
  for (int _u_i=1; _u_i<=134; _u_i++) {
    for (int _u_j=1; _u_j<=134; _u_j++) {
      double _tmpxx37 = _u_U[_u_i-1][_u_j-1];
      _u_acc = (_u_acc+_tmpxx37);
    }
  }
  scilab_rt_disp_s0_("Accumulated sum of all elements of the array U:");
  scilab_rt_disp_d0_(_u_acc);

  scilab_rt_terminate();
}

