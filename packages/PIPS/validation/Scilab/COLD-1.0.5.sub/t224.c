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

  /*  t224.sce - from mcgill/crni.sce */
  /* This function finds the Crank-Nicholson solution */
  /* to the one-dimensional heat equation */
  /* tic(); */
  double _u_a = 2.5;
  double _u_b = 1.5;
  double _u_c = 5.0;
  int _u_m = 230;
  int _u_n = 230;
  int _tmpxx0 = (_u_n-1);
  double _u_h = (_u_a / _tmpxx0);
  int _tmpxx1 = (_u_m-1);
  double _u_k = (_u_b / _tmpxx1);
  double _tmpxx2 = pow(_u_c,2);
  double _tmpxx3 = (_tmpxx2*_u_k);
  double _tmpxx4 = pow(_u_h,2);
  double _u_r = (_tmpxx3 / _tmpxx4);
  double _tmpxx5 = (2 / _u_r);
  double _u_s1 = (2.0+_tmpxx5);
  double _tmpxx6 = (2.0 / _u_r);
  double _u_s2 = (_tmpxx6-2);
  double _u_U[230][230];
  scilab_rt_zeros_i0i0_d2(_u_n,_u_m,230,230,_u_U);
  for (int _u_i1=2; _u_i1<=(_u_n-1); _u_i1++) {
    double _tmpxx7 = (scilab_rt_sin_d0_(((SCILAB_PI*_u_h)*(_u_i1-1)))+scilab_rt_sin_d0_((((3*SCILAB_PI)*_u_h)*(_u_i1-1))));
    _u_U[_u_i1-1][0] = _tmpxx7;
  }
  double _tmpxx8[1][230];
  scilab_rt_ones_i0i0_d2(1,_u_n,1,230,_tmpxx8);
  double _u_Vd[1][230];
  scilab_rt_mul_d0d2_d2(_u_s1,1,230,_tmpxx8,1,230,_u_Vd);
  _u_Vd[0][(1-1)] = 1;
  _u_Vd[0][(_u_n-1)] = 1;
  double _tmpxx9[1][229];
  scilab_rt_ones_i0i0_d2(1,(_u_n-1),1,229,_tmpxx9);
  double _u_Va[1][229];
  scilab_rt_sub_d2_d2(1,229,_tmpxx9,1,229,_u_Va);
  _u_Va[0][((_u_n-1)-1)] = 0;
  double _tmpxx10[1][229];
  scilab_rt_ones_i0i0_d2(1,(_u_n-1),1,229,_tmpxx10);
  double _u_Vc[1][229];
  scilab_rt_sub_d2_d2(1,229,_tmpxx10,1,229,_u_Vc);
  _u_Vc[0][(1-1)] = 0;
  double _u_Vb[1][230];
  scilab_rt_ones_i0i0_d2(1,_u_n,1,230,_u_Vb);
  _u_Vb[0][(1-1)] = 0;
  _u_Vb[0][(_u_n-1)] = 0;
  for (int _u_j1=2; _u_j1<=_u_m; _u_j1++) {
    for (int _u_i1=2; _u_i1<=(_u_n-1); _u_i1++) {
      double _tmpxx11 = ((_u_U[(_u_i1-1)-1][(_u_j1-1)-1]+_u_U[(_u_i1+1)-1][(_u_j1-1)-1])+(_u_s2*_u_U[_u_i1-1][(_u_j1-1)-1]));
      _u_Vb[0][(_u_i1-1)] = _tmpxx11;
    }
    /* X = tridiagonal(Va, Vd, Vc, Vb); */
    double _u_X[1][230];
    scilab_rt_zeros_i0i0_d2(1,_u_n,1,230,_u_X);
    double _u_A[1][229];
    scilab_rt_assign_d2_d2(1,229,_u_Va,1,229,_u_A);
    double _u_D[1][230];
    scilab_rt_assign_d2_d2(1,230,_u_Vd,1,230,_u_D);
    double _u_C[1][229];
    scilab_rt_assign_d2_d2(1,229,_u_Vc,1,229,_u_C);
    double _u_B[1][230];
    scilab_rt_assign_d2_d2(1,230,_u_Vb,1,230,_u_B);
    int _u_s;
    scilab_rt_size_d2i0_i0(1,230,_u_B,2,&_u_s);
    for (int _u_l=2; _u_l<=_u_s; _u_l++) {
      double _tmpxx12 = _u_A[0][((_u_l-1)-1)];
      double _tmpxx13 = _u_D[0][((_u_l-1)-1)];
      double _u_mult = (_tmpxx12 / _tmpxx13);
      double _tmpxx14 = (_u_D[0][(_u_l-1)]-(_u_mult*_u_C[0][((_u_l-1)-1)]));
      _u_D[0][(_u_l-1)] = _tmpxx14;
      double _tmpxx15 = (_u_B[0][(_u_l-1)]-(_u_mult*_u_B[0][((_u_l-1)-1)]));
      _u_B[0][(_u_l-1)] = _tmpxx15;
    }
    double _tmpxx16 = (_u_B[0][(_u_s-1)] / _u_D[0][(_u_s-1)]);
    _u_X[0][(_u_s-1)] = _tmpxx16;
    for (int _u_l=1; _u_l<=(_u_s-1); _u_l++) {
      double _tmpxx17 = ((_u_B[0][((((_u_s-1)-_u_l)+1)-1)]-(_u_C[0][((((_u_s-1)-_u_l)+1)-1)]*_u_X[0][(((((_u_s-1)-_u_l)+1)+1)-1)])) / _u_D[0][((((_u_s-1)-_u_l)+1)-1)]);
      _u_X[0][((((_u_s-1)-_u_l)+1)-1)] = _tmpxx17;
    }
    /* //////////////////////////////////////////////////// */
    /* U(1:n, j1) = X'; */
    for (int _u_i=1; _u_i<=_u_n; _u_i++) {
      _u_U[_u_i-1][_u_j1-1] = _u_X[0][(_u_i-1)];
    }
  }
  /* elapsedTime = toc(); */
  /* disp("Elapsed Time: "); */
  /* disp(elapsedTime); */
  scilab_rt_disp_s0_("Mean of matrix U");
  double _tmpxx18;
  scilab_rt_mean_d2_d0(230,230,_u_U,&_tmpxx18);
  scilab_rt_disp_d0_(_tmpxx18);

  scilab_rt_terminate();
}

