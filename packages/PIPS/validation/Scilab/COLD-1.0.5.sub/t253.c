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

  /*  t253.sce: element wise multiplication with complex */
  int _u_as = 2;
  scilab_rt_display_s0i0_("as",_u_as);
  int _tmp0 = (_u_as*3);
  scilab_rt_display_s0i0_("ans",_tmp0);
  double complex _tmpxx0 = (3*I);
  double complex _u_bs = (3+_tmpxx0);
  scilab_rt_display_s0z0_("bs",_u_bs);
  double complex _tmp1 = (_u_bs*4);
  scilab_rt_display_s0z0_("ans",_tmp1);
  double complex _tmp2 = (5*_u_bs);
  scilab_rt_display_s0z0_("ans",_tmp2);
  double complex _tmpxx1 = (1*I);
  double complex _tmpxx2 = (2*I);
  double complex _u_a[1][2];
  _u_a[0][0]=(1+_tmpxx1);
  _u_a[0][1]=(2+_tmpxx2);
  scilab_rt_display_s0z2_("a",1,2,_u_a);
  double complex _tmp3[1][2];
  scilab_rt_eltmul_z2i0_z2(1,2,_u_a,10,1,2,_tmp3);
  scilab_rt_display_s0z2_("ans",1,2,_tmp3);
  double complex _tmpxx3[1][2];
  scilab_rt_eltmul_z2i0_z2(1,2,_u_a,10,1,2,_tmpxx3);
  double complex _tmpxx4 = (2*I);
  double complex _tmp4[1][2];
  scilab_rt_add_z2z0_z2(1,2,_tmpxx3,_tmpxx4,1,2,_tmp4);
  scilab_rt_display_s0z2_("ans",1,2,_tmp4);
  double complex _tmp5[1][2];
  scilab_rt_eltmul_z0z2_z2(I,1,2,_u_a,1,2,_tmp5);
  scilab_rt_display_s0z2_("ans",1,2,_tmp5);
  int _tmpxx5[3][3];
  _tmpxx5[0][0]=1;
  _tmpxx5[0][1]=2;
  _tmpxx5[0][2]=3;
  _tmpxx5[1][0]=4;
  _tmpxx5[1][1]=5;
  _tmpxx5[1][2]=6;
  _tmpxx5[2][0]=7;
  _tmpxx5[2][1]=8;
  _tmpxx5[2][2]=9;
  double complex _tmp6[3][3];
  scilab_rt_eltmul_i2z0_z2(3,3,_tmpxx5,I,3,3,_tmp6);
  scilab_rt_display_s0z2_("ans",3,3,_tmp6);
  double complex _tmpxx6 = (1*I);
  double complex _tmpxx7 = (2*I);
  double complex _tmpxx8[1][2];
  _tmpxx8[0][0]=(1+_tmpxx6);
  _tmpxx8[0][1]=(2+_tmpxx7);
  double complex _tmp7[1][2];
  scilab_rt_eltmul_z2i0_z2(1,2,_tmpxx8,100,1,2,_tmp7);
  scilab_rt_display_s0z2_("ans",1,2,_tmp7);
  int _tmpxx9[3][3];
  _tmpxx9[0][0]=1;
  _tmpxx9[0][1]=2;
  _tmpxx9[0][2]=3;
  _tmpxx9[1][0]=4;
  _tmpxx9[1][1]=5;
  _tmpxx9[1][2]=6;
  _tmpxx9[2][0]=7;
  _tmpxx9[2][1]=8;
  _tmpxx9[2][2]=9;
  int _tmpxx10[3][3];
  scilab_rt_eltmul_i2i0_i2(3,3,_tmpxx9,100,3,3,_tmpxx10);
  double complex _tmpxx11 = (200*I);
  double complex _tmp8[3][3];
  scilab_rt_add_i2z0_z2(3,3,_tmpxx10,_tmpxx11,3,3,_tmp8);
  scilab_rt_display_s0z2_("ans",3,3,_tmp8);
  double complex _tmpxx12 = (0.1*I);
  double complex _tmpxx13 = (0.2*I);
  double complex _tmpxx14 = (0.3*I);
  double complex _tmpxx15 = (0.4*I);
  double complex _tmpxx16 = (1*I);
  double complex _tmpxx17 = (2*I);
  double complex _tmpxx18 = (3*I);
  double complex _tmpxx19 = (4*I);
  double complex _tmpxx20[2][2];
  _tmpxx20[0][0]=(0.1+_tmpxx12);
  _tmpxx20[0][1]=(0.2+_tmpxx13);
  _tmpxx20[1][0]=(0.3+_tmpxx14);
  _tmpxx20[1][1]=(0.4+_tmpxx15);
  double complex _tmpxx21[2][2];
  _tmpxx21[0][0]=(1+_tmpxx16);
  _tmpxx21[0][1]=(2+_tmpxx17);
  _tmpxx21[1][0]=(3+_tmpxx18);
  _tmpxx21[1][1]=(4+_tmpxx19);
  double complex _tmp9[2][2];
  scilab_rt_eltmul_z2z2_z2(2,2,_tmpxx20,2,2,_tmpxx21,2,2,_tmp9);
  scilab_rt_display_s0z2_("ans",2,2,_tmp9);
  /*  3D */
  double _u_b[2][2][2];
  scilab_rt_ones_i0i0i0_d3(2,2,2,2,2,2,_u_b);
  double _tmpxx22[2][2][2];
  scilab_rt_add_d3i0_d3(2,2,2,_u_b,1,2,2,2,_tmpxx22);
  double complex _tmpxx23 = (2*I);
  double complex _u_c[2][2][2];
  scilab_rt_add_d3z0_z3(2,2,2,_tmpxx22,_tmpxx23,2,2,2,_u_c);
  scilab_rt_display_s0z3_("c",2,2,2,_u_c);
  double complex _tmpxx24 = (4*I);
  double complex _tmpxx25 = (3+_tmpxx24);
  double complex _tmpxx26[2][2][2];
  scilab_rt_eltmul_z0d3_z3(_tmpxx25,2,2,2,_u_b,2,2,2,_tmpxx26);
  double complex _u_d[2][2][2];
  scilab_rt_eltmul_z3z0_z3(2,2,2,_tmpxx26,I,2,2,2,_u_d);
  scilab_rt_display_s0z3_("d",2,2,2,_u_d);
  double complex _tmpxx27 = (4*I);
  double complex _tmpxx28[2][2][2];
  scilab_rt_eltmul_z0d3_z3(_tmpxx27,2,2,2,_u_b,2,2,2,_tmpxx28);
  double complex _tmpxx29[2][2][2];
  scilab_rt_eltmul_z3z0_z3(2,2,2,_tmpxx28,I,2,2,2,_tmpxx29);
  double complex _u_e[2][2][2];
  scilab_rt_add_i0z3_z3(3,2,2,2,_tmpxx29,2,2,2,_u_e);
  scilab_rt_display_s0z3_("e",2,2,2,_u_e);
  double complex _tmpxx30 = (3*I);
  double _tmpxx31[2][2][2];
  scilab_rt_eltmul_d3d3_d3(2,2,2,_u_b,2,2,2,_u_b,2,2,2,_tmpxx31);
  double complex _tmpxx32 = (1+_tmpxx30);
  double complex _u_f[2][2][2];
  scilab_rt_eltmul_d3z0_z3(2,2,2,_tmpxx31,_tmpxx32,2,2,2,_u_f);
  scilab_rt_display_s0z3_("f",2,2,2,_u_f);

  scilab_rt_terminate();
}

