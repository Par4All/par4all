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

void foo_i2i2_(int _u_a_n0,int _u_a_n1,int _u_a[_u_a_n0][_u_a_n1], int _u_b_n0,int _u_b_n1,int _u_b[_u_b_n0][_u_b_n1])
{
  int _u_c[_u_b_n0][_u_b_n1];
  scilab_rt_add_i2i2_i2(_u_a_n0,_u_a_n1,_u_a,_u_b_n0,_u_b_n1,_u_b,_u_b_n0,_u_b_n1,_u_c);
}




/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t199.sce: dynamic arrays, user function */
  int _tmpxx0[1][2];
  _tmpxx0[0][0]=1;
  _tmpxx0[0][1]=2;
  int _tmpxx1[1][2];
  _tmpxx1[0][0]=3;
  _tmpxx1[0][1]=4;
  foo_i2i2_(1,2,_tmpxx0,1,2,_tmpxx1);
  int _tmpxx2[1][1];
  _tmpxx2[0][0]=1;
  int _tmpxx3[1][1];
  _tmpxx3[0][0]=3;
  foo_i2i2_(1,1,_tmpxx2,1,1,_tmpxx3);

  scilab_rt_terminate();
}

