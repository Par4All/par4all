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

  /*  t280.sce _ lsq function */
  int _u_a0[3][3];
  _u_a0[0][0]=1;
  _u_a0[0][1]=2;
  _u_a0[0][2]=3;
  _u_a0[1][0]=4;
  _u_a0[1][1]=5;
  _u_a0[1][2]=6;
  _u_a0[2][0]=7;
  _u_a0[2][1]=8;
  _u_a0[2][2]=9;
  scilab_rt_display_s0i2_("a0",3,3,_u_a0);
  int _u_b0[3][1];
  _u_b0[0][0]=4;
  _u_b0[1][0]=2;
  _u_b0[2][0]=8;
  scilab_rt_display_s0i2_("b0",3,1,_u_b0);
  double _u_x0[3][1];
  scilab_rt_lsq_i2i2_d2(3,3,_u_a0,3,1,_u_b0,3,1,_u_x0);
  scilab_rt_display_s0d2_("x0",3,1,_u_x0);
  double _u_r0[3][1];
  scilab_rt_mul_i2d2_d2(3,3,_u_a0,3,1,_u_x0,3,1,_u_r0);
  scilab_rt_display_s0d2_("r0",3,1,_u_r0);
  double _u_a1[3][3];
  _u_a1[0][0]=1.0;
  _u_a1[0][1]=2;
  _u_a1[0][2]=3;
  _u_a1[1][0]=4;
  _u_a1[1][1]=5;
  _u_a1[1][2]=6;
  _u_a1[2][0]=7;
  _u_a1[2][1]=8;
  _u_a1[2][2]=9;
  scilab_rt_display_s0d2_("a1",3,3,_u_a1);
  int _u_b1[3][2];
  _u_b1[0][0]=4;
  _u_b1[0][1]=8;
  _u_b1[1][0]=2;
  _u_b1[1][1]=3;
  _u_b1[2][0]=4;
  _u_b1[2][1]=9;
  scilab_rt_display_s0i2_("b1",3,2,_u_b1);
  double _u_x1[3][2];
  scilab_rt_lsq_d2i2_d2(3,3,_u_a1,3,2,_u_b1,3,2,_u_x1);
  scilab_rt_display_s0d2_("x1",3,2,_u_x1);
  double _u_r1[3][2];
  scilab_rt_mul_d2d2_d2(3,3,_u_a1,3,2,_u_x1,3,2,_u_r1);
  scilab_rt_display_s0d2_("r1",3,2,_u_r1);
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
  double _u_b2[2][3];
  _u_b2[0][0]=4.0;
  _u_b2[0][1]=8;
  _u_b2[0][2]=9;
  _u_b2[1][0]=5;
  _u_b2[1][1]=2;
  _u_b2[1][2]=3;
  scilab_rt_display_s0d2_("b2",2,3,_u_b2);
  double _u_x2[5][3];
  scilab_rt_lsq_i2d2_d2(2,5,_u_a2,2,3,_u_b2,5,3,_u_x2);
  scilab_rt_display_s0d2_("x2",5,3,_u_x2);
  double _u_r2[2][3];
  scilab_rt_mul_i2d2_d2(2,5,_u_a2,5,3,_u_x2,2,3,_u_r2);
  scilab_rt_display_s0d2_("r2",2,3,_u_r2);
  double _u_a3[4][2];
  _u_a3[0][0]=1.0;
  _u_a3[0][1]=2;
  _u_a3[1][0]=3;
  _u_a3[1][1]=4;
  _u_a3[2][0]=5;
  _u_a3[2][1]=6;
  _u_a3[3][0]=7;
  _u_a3[3][1]=8;
  scilab_rt_display_s0d2_("a3",4,2,_u_a3);
  double _u_b3[4][3];
  _u_b3[0][0]=4.0;
  _u_b3[0][1]=8;
  _u_b3[0][2]=9;
  _u_b3[1][0]=5;
  _u_b3[1][1]=2;
  _u_b3[1][2]=3;
  _u_b3[2][0]=2;
  _u_b3[2][1]=4;
  _u_b3[2][2]=9;
  _u_b3[3][0]=2;
  _u_b3[3][1]=1;
  _u_b3[3][2]=8;
  scilab_rt_display_s0d2_("b3",4,3,_u_b3);
  double _u_x3[2][3];
  scilab_rt_lsq_d2d2_d2(4,2,_u_a3,4,3,_u_b3,2,3,_u_x3);
  scilab_rt_display_s0d2_("x3",2,3,_u_x3);
  double _u_r3[4][3];
  scilab_rt_mul_d2d2_d2(4,2,_u_a3,2,3,_u_x3,4,3,_u_r3);
  scilab_rt_display_s0d2_("r3",4,3,_u_r3);

  scilab_rt_terminate();
}

