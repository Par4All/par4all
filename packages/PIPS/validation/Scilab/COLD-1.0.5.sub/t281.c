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

  /*  t281.sce _ lsq function with complex */
  double _u_a0[3][3];
  _u_a0[0][0]=1.0;
  _u_a0[0][1]=2;
  _u_a0[0][2]=3;
  _u_a0[1][0]=4;
  _u_a0[1][1]=5;
  _u_a0[1][2]=6;
  _u_a0[2][0]=7;
  _u_a0[2][1]=8;
  _u_a0[2][2]=9;
  scilab_rt_display_s0d2_("a0",3,3,_u_a0);
  double complex _tmpxx0 = (2*I);
  int _tmpxx1[3][1];
  _tmpxx1[0][0]=4;
  _tmpxx1[1][0]=2;
  _tmpxx1[2][0]=8;
  double complex _tmpxx2 = (1+_tmpxx0);
  double complex _u_b0[3][1];
  scilab_rt_mul_i2z0_z2(3,1,_tmpxx1,_tmpxx2,3,1,_u_b0);
  scilab_rt_display_s0z2_("b0",3,1,_u_b0);
  double complex _u_x0[3][1];
  scilab_rt_lsq_d2z2_z2(3,3,_u_a0,3,1,_u_b0,3,1,_u_x0);
  scilab_rt_display_s0z2_("x0",3,1,_u_x0);
  double complex _u_r0[3][1];
  scilab_rt_mul_d2z2_z2(3,3,_u_a0,3,1,_u_x0,3,1,_u_r0);
  scilab_rt_display_s0z2_("r0",3,1,_u_r0);
  double complex _tmpxx3 = (2*I);
  double _tmpxx4[3][3];
  _tmpxx4[0][0]=1.0;
  _tmpxx4[0][1]=2;
  _tmpxx4[0][2]=3;
  _tmpxx4[1][0]=4;
  _tmpxx4[1][1]=5;
  _tmpxx4[1][2]=6;
  _tmpxx4[2][0]=7;
  _tmpxx4[2][1]=8;
  _tmpxx4[2][2]=9;
  double complex _tmpxx5 = (1+_tmpxx3);
  double complex _u_a1[3][3];
  scilab_rt_mul_d2z0_z2(3,3,_tmpxx4,_tmpxx5,3,3,_u_a1);
  scilab_rt_display_s0z2_("a1",3,3,_u_a1);
  int _u_b1[3][2];
  _u_b1[0][0]=4;
  _u_b1[0][1]=8;
  _u_b1[1][0]=2;
  _u_b1[1][1]=3;
  _u_b1[2][0]=4;
  _u_b1[2][1]=9;
  scilab_rt_display_s0i2_("b1",3,2,_u_b1);
  double complex _u_x1[3][2];
  scilab_rt_lsq_z2i2_z2(3,3,_u_a1,3,2,_u_b1,3,2,_u_x1);
  scilab_rt_display_s0z2_("x1",3,2,_u_x1);
  double complex _u_r1[3][2];
  scilab_rt_mul_z2z2_z2(3,3,_u_a1,3,2,_u_x1,3,2,_u_r1);
  scilab_rt_display_s0z2_("r1",3,2,_u_r1);
  int _u_a2[2][5];
  _u_a2[0][0]=1;
  _u_a2[0][1]=2;
  _u_a2[0][2]=3;
  _u_a2[0][3]=4;
  _u_a2[0][4]=5;
  _u_a2[1][0]=6;
  _u_a2[1][1]=7;
  _u_a2[1][2]=8;
  _u_a2[1][3]=9;
  _u_a2[1][4]=10;
  scilab_rt_display_s0i2_("a2",2,5,_u_a2);
  double complex _tmpxx6 = (2*I);
  double _tmpxx7[2][3];
  _tmpxx7[0][0]=4.0;
  _tmpxx7[0][1]=8;
  _tmpxx7[0][2]=9;
  _tmpxx7[1][0]=5;
  _tmpxx7[1][1]=2;
  _tmpxx7[1][2]=3;
  double complex _tmpxx8 = (1+_tmpxx6);
  double complex _u_b2[2][3];
  scilab_rt_mul_d2z0_z2(2,3,_tmpxx7,_tmpxx8,2,3,_u_b2);
  scilab_rt_display_s0z2_("b2",2,3,_u_b2);
  double complex _u_x2[5][3];
  scilab_rt_lsq_i2z2_z2(2,5,_u_a2,2,3,_u_b2,5,3,_u_x2);
  scilab_rt_display_s0z2_("x2",5,3,_u_x2);
  double complex _u_r2[2][3];
  scilab_rt_mul_i2z2_z2(2,5,_u_a2,5,3,_u_x2,2,3,_u_r2);
  scilab_rt_display_s0z2_("r2",2,3,_u_r2);
  double complex _tmpxx9 = (2*I);
  double _tmpxx10[4][2];
  _tmpxx10[0][0]=1.0;
  _tmpxx10[0][1]=2;
  _tmpxx10[1][0]=3;
  _tmpxx10[1][1]=4;
  _tmpxx10[2][0]=5;
  _tmpxx10[2][1]=6;
  _tmpxx10[3][0]=7;
  _tmpxx10[3][1]=8;
  double complex _tmpxx11 = (1+_tmpxx9);
  double complex _u_a3[4][2];
  scilab_rt_mul_d2z0_z2(4,2,_tmpxx10,_tmpxx11,4,2,_u_a3);
  scilab_rt_display_s0z2_("a3",4,2,_u_a3);
  double complex _tmpxx12 = (2*I);
  double _tmpxx13[4][3];
  _tmpxx13[0][0]=4.0;
  _tmpxx13[0][1]=8;
  _tmpxx13[0][2]=9;
  _tmpxx13[1][0]=5;
  _tmpxx13[1][1]=2;
  _tmpxx13[1][2]=3;
  _tmpxx13[2][0]=2;
  _tmpxx13[2][1]=4;
  _tmpxx13[2][2]=9;
  _tmpxx13[3][0]=2;
  _tmpxx13[3][1]=1;
  _tmpxx13[3][2]=8;
  double complex _tmpxx14 = (1+_tmpxx12);
  double complex _u_b3[4][3];
  scilab_rt_mul_d2z0_z2(4,3,_tmpxx13,_tmpxx14,4,3,_u_b3);
  scilab_rt_display_s0z2_("b3",4,3,_u_b3);
  double complex _u_x3[2][3];
  scilab_rt_lsq_z2z2_z2(4,2,_u_a3,4,3,_u_b3,2,3,_u_x3);
  scilab_rt_display_s0z2_("x3",2,3,_u_x3);
  double complex _u_r3[4][3];
  scilab_rt_mul_z2z2_z2(4,2,_u_a3,2,3,_u_x3,4,3,_u_r3);
  scilab_rt_display_s0z2_("r3",4,3,_u_r3);

  scilab_rt_terminate();
}

