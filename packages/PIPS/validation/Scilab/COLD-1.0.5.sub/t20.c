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

  double _u_a[10][10];
  scilab_rt_ones_i0i0_d2(10,10,10,10,_u_a);
  double _tmpxx0[10][10];
  scilab_rt_add_d2d2_d2(10,10,_u_a,10,10,_u_a,10,10,_tmpxx0);
  double _u_b[10][10];
  scilab_rt_add_d2d2_d2(10,10,_tmpxx0,10,10,_u_a,10,10,_u_b);
  scilab_rt_display_s0d2_("b",10,10,_u_b);
  double _tmpxx1[10][10];
  scilab_rt_mul_d2d0_d2(10,10,_u_b,2.,10,10,_tmpxx1);
  double _u_c[10][10];
  scilab_rt_add_d2d0_d2(10,10,_tmpxx1,3.,10,10,_u_c);
  scilab_rt_display_s0d2_("c",10,10,_u_c);
  double _tmpxx2[10][10];
  scilab_rt_add_d2d2_d2(10,10,_u_a,10,10,_u_b,10,10,_tmpxx2);
  double _tmpxx3[10][10];
  scilab_rt_add_d2i0_d2(10,10,_u_c,2,10,10,_tmpxx3);
  double _u_d[10][10];
  scilab_rt_mul_d2d2_d2(10,10,_tmpxx2,10,10,_tmpxx3,10,10,_u_d);
  scilab_rt_display_s0d2_("d",10,10,_u_d);

  scilab_rt_terminate();
}

