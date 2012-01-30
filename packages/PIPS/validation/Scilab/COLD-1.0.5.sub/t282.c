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

  /*  t282.sce _ schur function */
  int _u_a1[3][3];
  _u_a1[0][0]=1;
  _u_a1[0][1]=2;
  _u_a1[0][2]=3;
  _u_a1[1][0]=4;
  _u_a1[1][1]=5;
  _u_a1[1][2]=6;
  _u_a1[2][0]=7;
  _u_a1[2][1]=8;
  _u_a1[2][2]=9;
  scilab_rt_display_s0i2_("a1",3,3,_u_a1);
  double _u_u1[3][3];
  double _u_t1[3][3];
  scilab_rt_schur_i2_d2d2(3,3,_u_a1,3,3,_u_u1,3,3,_u_t1);
  scilab_rt_display_s0d2_("t1",3,3,_u_t1);
  scilab_rt_display_s0d2_("u1",3,3,_u_u1);
  double _tmpxx0[3][3];
  scilab_rt_mul_d2d2_d2(3,3,_u_u1,3,3,_u_t1,3,3,_tmpxx0);
  double _tmpxx1[3][3];
  scilab_rt_transposeConjugate_d2_d2(3,3,_u_u1,3,3,_tmpxx1);
  double _u_b1[3][3];
  scilab_rt_mul_d2d2_d2(3,3,_tmpxx0,3,3,_tmpxx1,3,3,_u_b1);
  scilab_rt_display_s0d2_("b1",3,3,_u_b1);
  double _u_a2[5][5];
  _u_a2[0][0]=2.0;
  _u_a2[0][1]=3;
  _u_a2[0][2]=8;
  _u_a2[0][3]=1;
  _u_a2[0][4]=9;
  _u_a2[1][0]=5;
  _u_a2[1][1]=3;
  _u_a2[1][2]=4;
  _u_a2[1][3]=9;
  _u_a2[1][4]=7;
  _u_a2[2][0]=2;
  _u_a2[2][1]=4;
  _u_a2[2][2]=6;
  _u_a2[2][3]=3;
  _u_a2[2][4]=7;
  _u_a2[3][0]=1;
  _u_a2[3][1]=3;
  _u_a2[3][2]=2;
  _u_a2[3][3]=8;
  _u_a2[3][4]=4;
  _u_a2[4][0]=1;
  _u_a2[4][1]=3;
  _u_a2[4][2]=2;
  _u_a2[4][3]=5;
  _u_a2[4][4]=8;
  scilab_rt_display_s0d2_("a2",5,5,_u_a2);
  double _u_u2[5][5];
  double _u_t2[5][5];
  scilab_rt_schur_d2_d2d2(5,5,_u_a2,5,5,_u_u2,5,5,_u_t2);
  scilab_rt_display_s0d2_("t2",5,5,_u_t2);
  scilab_rt_display_s0d2_("u2",5,5,_u_u2);
  double _tmpxx2[5][5];
  scilab_rt_mul_d2d2_d2(5,5,_u_u2,5,5,_u_t2,5,5,_tmpxx2);
  double _tmpxx3[5][5];
  scilab_rt_transposeConjugate_d2_d2(5,5,_u_u2,5,5,_tmpxx3);
  double _u_b2[5][5];
  scilab_rt_mul_d2d2_d2(5,5,_tmpxx2,5,5,_tmpxx3,5,5,_u_b2);
  scilab_rt_display_s0d2_("b2",5,5,_u_b2);
  double complex _tmpxx4 = (2*I);
  int _tmpxx5[4][4];
  _tmpxx5[0][0]=2;
  _tmpxx5[0][1]=3;
  _tmpxx5[0][2]=8;
  _tmpxx5[0][3]=1;
  _tmpxx5[1][0]=2;
  _tmpxx5[1][1]=4;
  _tmpxx5[1][2]=6;
  _tmpxx5[1][3]=3;
  _tmpxx5[2][0]=1;
  _tmpxx5[2][1]=3;
  _tmpxx5[2][2]=2;
  _tmpxx5[2][3]=8;
  _tmpxx5[3][0]=1;
  _tmpxx5[3][1]=3;
  _tmpxx5[3][2]=2;
  _tmpxx5[3][3]=5;
  double complex _tmpxx6 = (1+_tmpxx4);
  double complex _u_a3[4][4];
  scilab_rt_mul_i2z0_z2(4,4,_tmpxx5,_tmpxx6,4,4,_u_a3);
  scilab_rt_display_s0z2_("a3",4,4,_u_a3);
  double complex _u_u3[4][4];
  double complex _u_t3[4][4];
  scilab_rt_schur_z2_z2z2(4,4,_u_a3,4,4,_u_u3,4,4,_u_t3);
  scilab_rt_display_s0z2_("t3",4,4,_u_t3);
  scilab_rt_display_s0z2_("u3",4,4,_u_u3);
  double complex _tmpxx7[4][4];
  scilab_rt_mul_z2z2_z2(4,4,_u_u3,4,4,_u_t3,4,4,_tmpxx7);
  double complex _tmpxx8[4][4];
  scilab_rt_transposeConjugate_z2_z2(4,4,_u_u3,4,4,_tmpxx8);
  double complex _u_b3[4][4];
  scilab_rt_mul_z2z2_z2(4,4,_tmpxx7,4,4,_tmpxx8,4,4,_u_b3);
  scilab_rt_display_s0z2_("b3",4,4,_u_b3);

  scilab_rt_terminate();
}

