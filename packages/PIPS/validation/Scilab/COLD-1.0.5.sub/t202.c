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

void foo_i0_d2(int _u_N, int _u_H_n0,int _u_H_n1,double _u_H[_u_H_n0][_u_H_n1])
{
  
  scilab_rt_zeros_i0i0_d2(_u_N,_u_N,_u_H_n0,_u_H_n1,_u_H);
  for (int _u_j=2; _u_j<=(_u_N-1); _u_j++) {
    _u_H[_u_j-1][_u_j-1] = _u_j;
  }
  int _tmpxx1 = (3*_u_N);
  _u_H[_u_N-1][_u_N-1] = _tmpxx1;
  int _tmpxx2 = (3*_u_N);
  _u_H[0][0] = _tmpxx2;
  for (int _u_j=2; _u_j<=_u_N; _u_j++) {
    int _tmpxx3 = (-2);
    _u_H[(_u_j-1)-1][_u_j-1] = _tmpxx3;
    int _tmpxx4 = (-2);
    _u_H[_u_j-1][(_u_j-1)-1] = _tmpxx4;
  }
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t202.sce: RT spec, gsort */
  scilab_rt_lines_d0_(0);
  double _u_H[10][10];
  foo_i0_d2(10,10,10,_u_H);
  double _u_Y[10][10];
  double _u_D[10][10];
  scilab_rt_spec_d2_d2d2(10,10,_u_H,10,10,_u_Y,10,10,_u_D);
  double _tmpxx0[10][1];
  scilab_rt_diag_d2_d2(10,10,_u_D,10,1,_tmpxx0);
  double _u_lambda1[10][1];
  int _u_key1[10][1];
  scilab_rt_gsort_d2s0s0_d2i2(10,1,_tmpxx0,"g","i",10,1,_u_lambda1,10,1,_u_key1);
  scilab_rt_disp_d2_(10,1,_u_lambda1);
  scilab_rt_disp_i2_(10,1,_u_key1);

  scilab_rt_terminate();
}

