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

double foo_i0i2i0_(int _u_s, int _u_sA_n0,int _u_sA_n1,int _u_sA[_u_sA_n0][_u_sA_n1], int _u_p)
{
  int _tmpxx0[_u_sA_n0][_u_sA_n1];
  scilab_rt_abs_i2_i2(_u_sA_n0,_u_sA_n1,_u_sA,_u_sA_n0,_u_sA_n1,_tmpxx0);
  int _tmpxx1[_u_sA_n0][_u_sA_n1];
  scilab_rt_eltpow_i2i0_i2(_u_sA_n0,_u_sA_n1,_tmpxx0,_u_p,_u_sA_n0,_u_sA_n1,_tmpxx1);
  int _tmpxx2;
  scilab_rt_sum_i2_i0(_u_sA_n0,_u_sA_n1,_tmpxx1,&_tmpxx2);
  double _tmpxx3 = ((double)1 / _u_p);
  double _tmpxx4 = pow(_tmpxx2,_tmpxx3);
  double _u_y = (_u_s*_tmpxx4);
  return _u_y;
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t204.sce: CODEGEN for A = call(), user function */
  int _u_s = 4;
  int _u_sA[1][4];
  _u_sA[0][0]=1;
  _u_sA[0][1]=2;
  _u_sA[0][2]=3;
  _u_sA[0][3]=4;
  int _u_p = 2;
  double _tmp0 = foo_i0i2i0_(_u_s,1,4,_u_sA,_u_p);
  scilab_rt_display_s0d0_("ans",_tmp0);

  scilab_rt_terminate();
}

