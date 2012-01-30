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

  /*  t262.sce: power with complex */
  int _u_as = 2;
  scilab_rt_display_s0i0_("as",_u_as);
  int _tmp0 = pow(_u_as,3);
  scilab_rt_display_s0i0_("ans",_tmp0);
  int _tmp1 = pow(3,_u_as);
  scilab_rt_display_s0i0_("ans",_tmp1);
  double complex _tmpxx0 = (3*I);
  double complex _u_bs = (3+_tmpxx0);
  scilab_rt_display_s0z0_("bs",_u_bs);
  double complex _tmp2 = cpow(_u_bs,5);
  scilab_rt_display_s0z0_("ans",_tmp2);
  double complex _tmp3 = cpow(5,_u_bs);
  scilab_rt_display_s0z0_("ans",_tmp3);
  double complex _tmpxx1 = (1*I);
  double complex _tmpxx2 = (2*I);
  double complex _u_a[1][2];
  _u_a[0][0]=(1+_tmpxx1);
  _u_a[0][1]=(2+_tmpxx2);
  scilab_rt_display_s0z2_("a",1,2,_u_a);
  double complex _tmpxx3 = (1*I);
  double complex _tmpxx4 = (2*I);
  double complex _tmpxx5 = (4*I);
  double complex _tmpxx6 = (6*I);
  double complex _u_b[2][2];
  _u_b[0][0]=(1+_tmpxx3);
  _u_b[0][1]=(2+_tmpxx4);
  _u_b[1][0]=(3+_tmpxx5);
  _u_b[1][1]=(5+_tmpxx6);
  scilab_rt_display_s0z2_("b",2,2,_u_b);
  double complex _tmp4[1][2];
  scilab_rt_pow_z2i0_z2(1,2,_u_a,10,1,2,_tmp4);
  scilab_rt_display_s0z2_("ans",1,2,_tmp4);
  double complex _tmpxx7[1][2];
  scilab_rt_pow_z2i0_z2(1,2,_u_a,10,1,2,_tmpxx7);
  double complex _tmpxx8 = (2*I);
  double complex _tmp5[1][2];
  scilab_rt_add_z2z0_z2(1,2,_tmpxx7,_tmpxx8,1,2,_tmp5);
  scilab_rt_display_s0z2_("ans",1,2,_tmp5);
  double complex _tmp6[1][2];
  scilab_rt_pow_z0z2_z2(I,1,2,_u_a,1,2,_tmp6);
  scilab_rt_display_s0z2_("ans",1,2,_tmp6);
  double complex _tmp7[2][2];
  scilab_rt_pow_z0z2_z2(I,2,2,_u_b,2,2,_tmp7);
  scilab_rt_display_s0z2_("ans",2,2,_tmp7);
  double complex _tmpxx9 = (1*I);
  double complex _tmpxx10 = (2*I);
  double complex _tmpxx11[1][2];
  _tmpxx11[0][0]=(1+_tmpxx9);
  _tmpxx11[0][1]=(2+_tmpxx10);
  double complex _tmp8[1][2];
  scilab_rt_pow_z2i0_z2(1,2,_tmpxx11,2,1,2,_tmp8);
  scilab_rt_display_s0z2_("ans",1,2,_tmp8);
  int _tmpxx12[1][9];
  _tmpxx12[0][0]=1;
  _tmpxx12[0][1]=2;
  _tmpxx12[0][2]=3;
  _tmpxx12[0][3]=4;
  _tmpxx12[0][4]=5;
  _tmpxx12[0][5]=6;
  _tmpxx12[0][6]=7;
  _tmpxx12[0][7]=8;
  _tmpxx12[0][8]=9;
  int _tmpxx13[1][9];
  scilab_rt_pow_i2i0_i2(1,9,_tmpxx12,1,1,9,_tmpxx13);
  double complex _tmpxx14 = (2*I);
  double complex _tmp9[1][9];
  scilab_rt_add_i2z0_z2(1,9,_tmpxx13,_tmpxx14,1,9,_tmp9);
  scilab_rt_display_s0z2_("ans",1,9,_tmp9);

  scilab_rt_terminate();
}

