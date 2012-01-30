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

  /*  t256.ce: conjugate transpose and no conjugate transpose with complex */
  int _u_a[2][4];
  _u_a[0][0]=1;
  _u_a[0][1]=2;
  _u_a[0][2]=3;
  _u_a[0][3]=4;
  _u_a[1][0]=4;
  _u_a[1][1]=5;
  _u_a[1][2]=6;
  _u_a[1][3]=7;
  scilab_rt_display_s0i2_("a",2,4,_u_a);
  int _u_acT[4][2];
  scilab_rt_transposeConjugate_i2_i2(2,4,_u_a,4,2,_u_acT);
  scilab_rt_display_s0i2_("acT",4,2,_u_acT);
  int _u_anT[4][2];
  scilab_rt_transpose_i2_i2(2,4,_u_a,4,2,_u_anT);
  scilab_rt_display_s0i2_("anT",4,2,_u_anT);
  double complex _tmpxx0 = (2*I);
  double complex _tmpxx1 = (4*I);
  double complex _tmpxx2 = (6*I);
  double complex _tmpxx3 = (8*I);
  double complex _u_b[2][2];
  _u_b[0][0]=(1+_tmpxx0);
  _u_b[0][1]=(3+_tmpxx1);
  _u_b[1][0]=(5+_tmpxx2);
  _u_b[1][1]=(7+_tmpxx3);
  scilab_rt_display_s0z2_("b",2,2,_u_b);
  double complex _u_bcT[2][2];
  scilab_rt_transposeConjugate_z2_z2(2,2,_u_b,2,2,_u_bcT);
  scilab_rt_display_s0z2_("bcT",2,2,_u_bcT);
  double complex _u_bnT[2][2];
  scilab_rt_transpose_z2_z2(2,2,_u_b,2,2,_u_bnT);
  scilab_rt_display_s0z2_("bnT",2,2,_u_bnT);
  double complex _tmpxx4 = (3*I);
  double _tmpxx5[3][4];
  scilab_rt_ones_i0i0_d2(3,4,3,4,_tmpxx5);
  double complex _tmpxx6 = (2+_tmpxx4);
  double complex _u_c[3][4];
  scilab_rt_mul_d2z0_z2(3,4,_tmpxx5,_tmpxx6,3,4,_u_c);
  scilab_rt_display_s0z2_("c",3,4,_u_c);
  double complex _u_ccT[4][3];
  scilab_rt_transposeConjugate_z2_z2(3,4,_u_c,4,3,_u_ccT);
  scilab_rt_display_s0z2_("ccT",4,3,_u_ccT);
  double complex _u_cnT[4][3];
  scilab_rt_transpose_z2_z2(3,4,_u_c,4,3,_u_cnT);
  scilab_rt_display_s0z2_("cnT",4,3,_u_cnT);

  scilab_rt_terminate();
}

