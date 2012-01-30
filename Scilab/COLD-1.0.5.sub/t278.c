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

  /*  t278.sce _ function qr */
  /* Square */
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
  double _u_Q1[3][3];
  double _u_R1[3][3];
  scilab_rt_qr_i2_d2d2(3,3,_u_a1,3,3,_u_Q1,3,3,_u_R1);
  scilab_rt_display_s0d2_("R1",3,3,_u_R1);
  scilab_rt_display_s0d2_("Q1",3,3,_u_Q1);
  scilab_rt_display_s0i2_("a1",3,3,_u_a1);
  double _u_b1[3][3];
  scilab_rt_mul_d2d2_d2(3,3,_u_Q1,3,3,_u_R1,3,3,_u_b1);
  scilab_rt_display_s0d2_("b1",3,3,_u_b1);
  /*  m > n */
  double _u_a2[2][5];
  _u_a2[0][0]=1.0;
  _u_a2[0][1]=2;
  _u_a2[0][2]=3;
  _u_a2[0][3]=4;
  _u_a2[0][4]=5;
  _u_a2[1][0]=6;
  _u_a2[1][1]=7;
  _u_a2[1][2]=8;
  _u_a2[1][3]=9;
  _u_a2[1][4]=10;
  double _u_Q2[2][2];
  double _u_R2[2][5];
  scilab_rt_qr_d2_d2d2(2,5,_u_a2,2,2,_u_Q2,2,5,_u_R2);
  scilab_rt_display_s0d2_("R2",2,5,_u_R2);
  scilab_rt_display_s0d2_("Q2",2,2,_u_Q2);
  scilab_rt_display_s0d2_("a2",2,5,_u_a2);
  double _u_b2[2][5];
  scilab_rt_mul_d2d2_d2(2,2,_u_Q2,2,5,_u_R2,2,5,_u_b2);
  scilab_rt_display_s0d2_("b2",2,5,_u_b2);
  /*  m < n */
  int _u_a3[4][2];
  _u_a3[0][0]=1;
  _u_a3[0][1]=2;
  _u_a3[1][0]=3;
  _u_a3[1][1]=4;
  _u_a3[2][0]=5;
  _u_a3[2][1]=6;
  _u_a3[3][0]=7;
  _u_a3[3][1]=8;
  double _u_Q3[4][4];
  double _u_R3[4][2];
  scilab_rt_qr_i2_d2d2(4,2,_u_a3,4,4,_u_Q3,4,2,_u_R3);
  scilab_rt_display_s0d2_("R3",4,2,_u_R3);
  scilab_rt_display_s0d2_("Q3",4,4,_u_Q3);
  scilab_rt_display_s0i2_("a3",4,2,_u_a3);
  double _u_b3[4][2];
  scilab_rt_mul_d2d2_d2(4,4,_u_Q3,4,2,_u_R3,4,2,_u_b3);
  scilab_rt_display_s0d2_("b3",4,2,_u_b3);

  scilab_rt_terminate();
}

