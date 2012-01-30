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

  /*  t67.sce: testing rand function */
  double _tmp0[3][3];
  scilab_rt_rand_i0i0_d2(3,3,3,3,_tmp0);
  scilab_rt_display_s0d2_("ans",3,3,_tmp0);
  double _tmp1[3][3];
  scilab_rt_rand_i0i0_d2(3,3,3,3,_tmp1);
  scilab_rt_display_s0d2_("ans",3,3,_tmp1);
  double _tmp2[3][3];
  scilab_rt_rand_i0i0_d2(3,3,3,3,_tmp2);
  scilab_rt_display_s0d2_("ans",3,3,_tmp2);
  int _u_a[3][3];
  _u_a[0][0]=1;
  _u_a[0][1]=2;
  _u_a[0][2]=3;
  _u_a[1][0]=4;
  _u_a[1][1]=5;
  _u_a[1][2]=6;
  _u_a[2][0]=7;
  _u_a[2][1]=8;
  _u_a[2][2]=9;
  double _tmp3[3][3];
  scilab_rt_rand_i2_d2(3,3,_u_a,3,3,_tmp3);
  scilab_rt_display_s0d2_("ans",3,3,_tmp3);
  double complex _tmpxx0 = (2*I);
  double complex _tmpxx1 = (4*I);
  double complex _tmpxx2 = (6*I);
  double complex _tmpxx3 = (8*I);
  double complex _u_b[2][2];
  _u_b[0][0]=(1+_tmpxx0);
  _u_b[0][1]=(3+_tmpxx1);
  _u_b[1][0]=(5+_tmpxx2);
  _u_b[1][1]=(7+_tmpxx3);
  double complex _tmp4[2][2];
  scilab_rt_rand_z2_z2(2,2,_u_b,2,2,_tmp4);
  scilab_rt_display_s0z2_("ans",2,2,_tmp4);
  double _tmp5[2][3][2];
  scilab_rt_rand_i0i0i0_d3(2,3,2,2,3,2,_tmp5);
  scilab_rt_display_s0d3_("ans",2,3,2,_tmp5);

  scilab_rt_terminate();
}

