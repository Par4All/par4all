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

  int _tmpxx0[1][2];
  _tmpxx0[0][0]=3;
  _tmpxx0[0][1]=4;
  int _tmpxx1[1][2];
  _tmpxx1[0][0]=1;
  _tmpxx1[0][1]=2;
  int _tmpxx2[1][2];
  scilab_rt_mul_i2i0_i2(1,2,_tmpxx0,2,1,2,_tmpxx2);
  int _u_a[1][2];
  scilab_rt_add_i2i2_i2(1,2,_tmpxx1,1,2,_tmpxx2,1,2,_u_a);
  scilab_rt_display_s0i2_("a",1,2,_u_a);
  int _tmpxx3[1][2];
  _tmpxx3[0][0]=7;
  _tmpxx3[0][1]=10;
  int _tmpif0[1][2];
  scilab_rt_ne_i2i2_i2(1,2,_u_a,1,2,_tmpxx3,1,2,_tmpif0);
  int _tmpif1;
  scilab_rt_reduce_to_bool_i2_i0(1,2,_tmpif0,&_tmpif1);
  if (_tmpif1) {
    scilab_rt_exit_i0_(1);
  }

  scilab_rt_terminate();
}

