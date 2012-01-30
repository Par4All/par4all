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

void foo_i2_(int _u_x_n0,int _u_x_n1,int _u_x[_u_x_n0][_u_x_n1])
{
  int _u_a[_u_x_n0][_u_x_n1];
  scilab_rt_add_i2i0_i2(_u_x_n0,_u_x_n1,_u_x,1,_u_x_n0,_u_x_n1,_u_a);
  scilab_rt_disp_i2_(_u_x_n0,_u_x_n1,_u_a);
}




/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t206.sce: user function */
  int _tmpxx0[1][2];
  _tmpxx0[0][0]=1;
  _tmpxx0[0][1]=2;
  foo_i2_(1,2,_tmpxx0);
  int _tmpxx1[1][4];
  _tmpxx1[0][0]=1;
  _tmpxx1[0][1]=2;
  _tmpxx1[0][2]=3;
  _tmpxx1[0][3]=4;
  foo_i2_(1,4,_tmpxx1);

  scilab_rt_terminate();
}

