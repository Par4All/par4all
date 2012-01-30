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

int foo_i0_(int _u_b)
{
  int _u_a = (_u_b+1);
  return _u_a;
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /* Testing x = x*x, x = x + y, n = foo(n), etc */
  double _u_x[3][3];
  _u_x[0][0]=1;
  _u_x[0][1]=2;
  _u_x[0][2]=3;
  _u_x[1][0]=4;
  _u_x[1][1]=5;
  _u_x[1][2]=6;
  _u_x[2][0]=7;
  _u_x[2][1]=8;
  _u_x[2][2]=9;
  scilab_rt_display_s0d2_("x",3,3,_u_x);
  double _u_y[3][3];
  scilab_rt_ones_i0i0_d2(3,3,3,3,_u_y);
  scilab_rt_display_s0d2_("y",3,3,_u_y);
  double _tmpxx0[3][3];
  scilab_rt_add_d2d2_d2(3,3,_u_x,3,3,_u_y,3,3,_tmpxx0);
  
  scilab_rt_assign_d2_d2(3,3,_tmpxx0,3,3,_u_x);
  scilab_rt_display_s0d2_("x",3,3,_u_x);
  double _tmpxx1[3][3];
  scilab_rt_mul_d2d2_d2(3,3,_u_x,3,3,_u_x,3,3,_tmpxx1);
  
  scilab_rt_assign_d2_d2(3,3,_tmpxx1,3,3,_u_x);
  scilab_rt_display_s0d2_("x",3,3,_u_x);
  int _u_n = 5;
  scilab_rt_display_s0i0_("n",_u_n);
  _u_n = foo_i0_(_u_n);
  scilab_rt_display_s0i0_("n",_u_n);

  scilab_rt_terminate();
}

