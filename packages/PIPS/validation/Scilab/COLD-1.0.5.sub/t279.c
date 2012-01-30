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

  /*  t278.sce _ function qr with complex */
  /* Square */
  double complex _tmpxx0 = (1*I);
  int _tmpxx1[3][3];
  _tmpxx1[0][0]=1;
  _tmpxx1[0][1]=2;
  _tmpxx1[0][2]=3;
  _tmpxx1[1][0]=4;
  _tmpxx1[1][1]=5;
  _tmpxx1[1][2]=6;
  _tmpxx1[2][0]=7;
  _tmpxx1[2][1]=8;
  _tmpxx1[2][2]=9;
  double complex _tmpxx2 = (1+_tmpxx0);
  double complex _u_a1[3][3];
  scilab_rt_mul_i2z0_z2(3,3,_tmpxx1,_tmpxx2,3,3,_u_a1);
  double complex _u_Q1[3][3];
  double complex _u_R1[3][3];
  scilab_rt_qr_z2_z2z2(3,3,_u_a1,3,3,_u_Q1,3,3,_u_R1);
  scilab_rt_display_s0z2_("R1",3,3,_u_R1);
  scilab_rt_display_s0z2_("Q1",3,3,_u_Q1);
  scilab_rt_display_s0z2_("a1",3,3,_u_a1);
  double complex _u_b1[3][3];
  scilab_rt_mul_z2z2_z2(3,3,_u_Q1,3,3,_u_R1,3,3,_u_b1);
  scilab_rt_display_s0z2_("b1",3,3,_u_b1);
  /*  m > n */
  double complex _tmpxx3 = (1*I);
  double _tmpxx4[2][5];
  _tmpxx4[0][0]=1.0;
  _tmpxx4[0][1]=2;
  _tmpxx4[0][2]=3;
  _tmpxx4[0][3]=4;
  _tmpxx4[0][4]=5;
  _tmpxx4[1][0]=6;
  _tmpxx4[1][1]=7;
  _tmpxx4[1][2]=8;
  _tmpxx4[1][3]=9;
  _tmpxx4[1][4]=10;
  double complex _tmpxx5 = (1+_tmpxx3);
  double complex _u_a2[2][5];
  scilab_rt_mul_d2z0_z2(2,5,_tmpxx4,_tmpxx5,2,5,_u_a2);
  double complex _u_Q2[2][2];
  double complex _u_R2[2][5];
  scilab_rt_qr_z2_z2z2(2,5,_u_a2,2,2,_u_Q2,2,5,_u_R2);
  scilab_rt_display_s0z2_("R2",2,5,_u_R2);
  scilab_rt_display_s0z2_("Q2",2,2,_u_Q2);
  scilab_rt_display_s0z2_("a2",2,5,_u_a2);
  double complex _u_b2[2][5];
  scilab_rt_mul_z2z2_z2(2,2,_u_Q2,2,5,_u_R2,2,5,_u_b2);
  scilab_rt_display_s0z2_("b2",2,5,_u_b2);
  /*  m < n */
  double complex _tmpxx6 = (1*I);
  int _tmpxx7[4][2];
  _tmpxx7[0][0]=1;
  _tmpxx7[0][1]=2;
  _tmpxx7[1][0]=3;
  _tmpxx7[1][1]=4;
  _tmpxx7[2][0]=5;
  _tmpxx7[2][1]=6;
  _tmpxx7[3][0]=7;
  _tmpxx7[3][1]=8;
  double complex _tmpxx8 = (1+_tmpxx6);
  double complex _u_a3[4][2];
  scilab_rt_mul_i2z0_z2(4,2,_tmpxx7,_tmpxx8,4,2,_u_a3);
  double complex _u_Q3[4][4];
  double complex _u_R3[4][2];
  scilab_rt_qr_z2_z2z2(4,2,_u_a3,4,4,_u_Q3,4,2,_u_R3);
  scilab_rt_display_s0z2_("R3",4,2,_u_R3);
  scilab_rt_display_s0z2_("Q3",4,4,_u_Q3);
  scilab_rt_display_s0z2_("a3",4,2,_u_a3);
  double complex _u_b3[4][2];
  scilab_rt_mul_z2z2_z2(4,4,_u_Q3,4,2,_u_R3,4,2,_u_b3);
  scilab_rt_display_s0z2_("b3",4,2,_u_b3);

  scilab_rt_terminate();
}

