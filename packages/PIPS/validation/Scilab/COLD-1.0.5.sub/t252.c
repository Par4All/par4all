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

  /*  t252.sce: multiplication with complex */
  double complex _tmpxx0 = (1*I);
  double complex _tmpxx1 = (2*I);
  double complex _u_a[1][2];
  _u_a[0][0]=(1+_tmpxx0);
  _u_a[0][1]=(2+_tmpxx1);
  scilab_rt_display_s0z2_("a",1,2,_u_a);
  double complex _tmp0[1][2];
  scilab_rt_mul_z2i0_z2(1,2,_u_a,10,1,2,_tmp0);
  scilab_rt_display_s0z2_("ans",1,2,_tmp0);
  double complex _tmpxx2[1][2];
  scilab_rt_mul_z2i0_z2(1,2,_u_a,10,1,2,_tmpxx2);
  double complex _tmpxx3 = (2*I);
  double complex _tmp1[1][2];
  scilab_rt_add_z2z0_z2(1,2,_tmpxx2,_tmpxx3,1,2,_tmp1);
  scilab_rt_display_s0z2_("ans",1,2,_tmp1);
  double complex _tmp2[1][2];
  scilab_rt_mul_z0z2_z2(I,1,2,_u_a,1,2,_tmp2);
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
  scilab_rt_mul_i2z0_z2(3,3,_tmpxx4,I,3,3,_tmp3);
  scilab_rt_display_s0z2_("ans",3,3,_tmp3);
  double complex _tmpxx5 = (1*I);
  double complex _tmpxx6 = (2*I);
  double complex _tmpxx7[1][2];
  _tmpxx7[0][0]=(1+_tmpxx5);
  _tmpxx7[0][1]=(2+_tmpxx6);
  double complex _tmp4[1][2];
  scilab_rt_mul_z2i0_z2(1,2,_tmpxx7,100,1,2,_tmp4);
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
  scilab_rt_mul_i2i0_i2(3,3,_tmpxx8,100,3,3,_tmpxx9);
  double complex _tmpxx10 = (200*I);
  double complex _tmp5[3][3];
  scilab_rt_add_i2z0_z2(3,3,_tmpxx9,_tmpxx10,3,3,_tmp5);
  scilab_rt_display_s0z2_("ans",3,3,_tmp5);
  double complex _tmpxx11 = (2.1*I);
  double complex _tmpxx12 = (4.2*I);
  double complex _tmpxx13 = (6.3*I);
  double complex _tmpxx14 = (8.4*I);
  double complex _tmpxx15 = (1*I);
  double complex _tmpxx16 = (2*I);
  double complex _tmpxx17 = (3*I);
  double complex _tmpxx18 = (4*I);
  double complex _tmpxx19[2][2];
  _tmpxx19[0][0]=(1.1+_tmpxx11);
  _tmpxx19[0][1]=(3.2+_tmpxx12);
  _tmpxx19[1][0]=(5.3+_tmpxx13);
  _tmpxx19[1][1]=(7.4+_tmpxx14);
  double complex _tmpxx20[2][2];
  _tmpxx20[0][0]=(1+_tmpxx15);
  _tmpxx20[0][1]=(2+_tmpxx16);
  _tmpxx20[1][0]=(3+_tmpxx17);
  _tmpxx20[1][1]=(4+_tmpxx18);
  double complex _tmp6[2][2];
  scilab_rt_mul_z2z2_z2(2,2,_tmpxx19,2,2,_tmpxx20,2,2,_tmp6);
  scilab_rt_display_s0z2_("ans",2,2,_tmp6);
  double _tmpxx21[2][4];
  scilab_rt_ones_i0i0_d2(2,4,2,4,_tmpxx21);
  double complex _u_x[2][4];
  scilab_rt_mul_d2z0_z2(2,4,_tmpxx21,I,2,4,_u_x);
  double _tmpxx22[4][5];
  scilab_rt_ones_i0i0_d2(4,5,4,5,_tmpxx22);
  double complex _u_y[4][5];
  scilab_rt_mul_d2z0_z2(4,5,_tmpxx22,I,4,5,_u_y);
  double complex _tmpxx23 = (3*I);
  double complex _tmpxx24[2][5];
  scilab_rt_mul_z2z2_z2(2,4,_u_x,4,5,_u_y,2,5,_tmpxx24);
  double complex _tmpxx25 = (2+_tmpxx23);
  double complex _u_z[2][5];
  scilab_rt_mul_z2z0_z2(2,5,_tmpxx24,_tmpxx25,2,5,_u_z);
  scilab_rt_display_s0z2_("z",2,5,_u_z);
  /*  3D */
  double _u_b[2][2][2];
  scilab_rt_ones_i0i0i0_d3(2,2,2,2,2,2,_u_b);
  double _tmpxx26[2][2][2];
  scilab_rt_add_d3i0_d3(2,2,2,_u_b,1,2,2,2,_tmpxx26);
  double complex _tmpxx27 = (2*I);
  double complex _u_c[2][2][2];
  scilab_rt_add_d3z0_z3(2,2,2,_tmpxx26,_tmpxx27,2,2,2,_u_c);
  scilab_rt_display_s0z3_("c",2,2,2,_u_c);
  double complex _tmpxx28 = (4*I);
  double complex _tmpxx29 = (3+_tmpxx28);
  double complex _tmpxx30[2][2][2];
  scilab_rt_mul_z0d3_z3(_tmpxx29,2,2,2,_u_b,2,2,2,_tmpxx30);
  double complex _u_d[2][2][2];
  scilab_rt_mul_z3z0_z3(2,2,2,_tmpxx30,I,2,2,2,_u_d);
  scilab_rt_display_s0z3_("d",2,2,2,_u_d);
  double complex _tmpxx31 = (4*I);
  double complex _tmpxx32[2][2][2];
  scilab_rt_mul_z0d3_z3(_tmpxx31,2,2,2,_u_b,2,2,2,_tmpxx32);
  double complex _tmpxx33[2][2][2];
  scilab_rt_mul_z3z0_z3(2,2,2,_tmpxx32,I,2,2,2,_tmpxx33);
  double complex _u_e[2][2][2];
  scilab_rt_add_i0z3_z3(3,2,2,2,_tmpxx33,2,2,2,_u_e);
  scilab_rt_display_s0z3_("e",2,2,2,_u_e);

  scilab_rt_terminate();
}

