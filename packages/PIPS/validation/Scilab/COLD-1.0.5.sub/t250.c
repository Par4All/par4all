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

  /*  t250.sce: addition with complex */
  double complex _tmpxx0 = (1*I);
  double complex _tmpxx1 = (2*I);
  double complex _u_a[1][2];
  _u_a[0][0]=(1+_tmpxx0);
  _u_a[0][1]=(2+_tmpxx1);
  scilab_rt_display_s0z2_("a",1,2,_u_a);
  double complex _tmp0[1][2];
  scilab_rt_add_z2i0_z2(1,2,_u_a,10,1,2,_tmp0);
  scilab_rt_display_s0z2_("ans",1,2,_tmp0);
  double complex _tmpxx2[1][2];
  scilab_rt_add_z2i0_z2(1,2,_u_a,10,1,2,_tmpxx2);
  double complex _tmpxx3 = (2*I);
  double complex _tmp1[1][2];
  scilab_rt_add_z2z0_z2(1,2,_tmpxx2,_tmpxx3,1,2,_tmp1);
  scilab_rt_display_s0z2_("ans",1,2,_tmp1);
  double complex _tmp2[1][2];
  scilab_rt_add_z0z2_z2(I,1,2,_u_a,1,2,_tmp2);
  scilab_rt_display_s0z2_("ans",1,2,_tmp2);
  int _tmpxx4[3][3];
  _tmpxx4[0][0]=1;
  _tmpxx4[0][1]=2;
  _tmpxx4[0][2]=3;
  _tmpxx4[1][0]=4;
  _tmpxx4[1][1]=5;
  _tmpxx4[1][2]=6;
  _tmpxx4[2][0]=7;
  _tmpxx4[2][1]=8;
  _tmpxx4[2][2]=9;
  double complex _tmp3[3][3];
  scilab_rt_add_i2z0_z2(3,3,_tmpxx4,I,3,3,_tmp3);
  scilab_rt_display_s0z2_("ans",3,3,_tmp3);
  double complex _tmpxx5 = (1*I);
  double complex _tmpxx6 = (2*I);
  double complex _tmpxx7[1][2];
  _tmpxx7[0][0]=(1+_tmpxx5);
  _tmpxx7[0][1]=(2+_tmpxx6);
  double complex _tmp4[1][2];
  scilab_rt_add_z2i0_z2(1,2,_tmpxx7,100,1,2,_tmp4);
  scilab_rt_display_s0z2_("ans",1,2,_tmp4);
  int _tmpxx8[3][3];
  _tmpxx8[0][0]=1;
  _tmpxx8[0][1]=2;
  _tmpxx8[0][2]=3;
  _tmpxx8[1][0]=4;
  _tmpxx8[1][1]=5;
  _tmpxx8[1][2]=6;
  _tmpxx8[2][0]=7;
  _tmpxx8[2][1]=8;
  _tmpxx8[2][2]=9;
  int _tmpxx9[3][3];
  scilab_rt_add_i2i0_i2(3,3,_tmpxx8,100,3,3,_tmpxx9);
  double complex _tmpxx10 = (200*I);
  double complex _tmp5[3][3];
  scilab_rt_add_i2z0_z2(3,3,_tmpxx9,_tmpxx10,3,3,_tmp5);
  scilab_rt_display_s0z2_("ans",3,3,_tmp5);
  double complex _tmpxx11 = (0.1*I);
  double complex _tmpxx12 = (0.2*I);
  double complex _tmpxx13 = (1*I);
  double complex _tmpxx14 = (2*I);
  double complex _tmpxx15[1][2];
  _tmpxx15[0][0]=(0.1+_tmpxx11);
  _tmpxx15[0][1]=(0.2+_tmpxx12);
  double complex _tmpxx16[1][2];
  _tmpxx16[0][0]=(1+_tmpxx13);
  _tmpxx16[0][1]=(2+_tmpxx14);
  double complex _tmp6[1][2];
  scilab_rt_add_z2z2_z2(1,2,_tmpxx15,1,2,_tmpxx16,1,2,_tmp6);
  scilab_rt_display_s0z2_("ans",1,2,_tmp6);
  /*  3D */
  double _u_b[2][2][2];
  scilab_rt_ones_i0i0i0_d3(2,2,2,2,2,2,_u_b);
  double _tmpxx17[2][2][2];
  scilab_rt_add_d3i0_d3(2,2,2,_u_b,1,2,2,2,_tmpxx17);
  double complex _tmpxx18 = (2*I);
  double complex _u_c[2][2][2];
  scilab_rt_add_d3z0_z3(2,2,2,_tmpxx17,_tmpxx18,2,2,2,_u_c);
  scilab_rt_display_s0z3_("c",2,2,2,_u_c);
  double complex _tmpxx19 = (4*I);
  double complex _tmpxx20 = (3+_tmpxx19);
  double complex _tmpxx21[2][2][2];
  scilab_rt_add_z0d3_z3(_tmpxx20,2,2,2,_u_b,2,2,2,_tmpxx21);
  double complex _u_d[2][2][2];
  scilab_rt_add_z3z0_z3(2,2,2,_tmpxx21,I,2,2,2,_u_d);
  scilab_rt_display_s0z3_("d",2,2,2,_u_d);
  double complex _tmp7[2][2][2];
  scilab_rt_add_z3z3_z3(2,2,2,_u_c,2,2,2,_u_d,2,2,2,_tmp7);
  scilab_rt_display_s0z3_("ans",2,2,2,_tmp7);

  scilab_rt_terminate();
}

