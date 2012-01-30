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

  /*  t269.sce _ array promotion */
  double _u_a[2][3];
  _u_a[0][0]=1;
  _u_a[0][1]=2;
  _u_a[0][2]=3;
  _u_a[1][0]=4;
  _u_a[1][1]=5;
  _u_a[1][2]=6;
  scilab_rt_display_s0d2_("a",2,3,_u_a);
  
  scilab_rt_ones_i0i0_d2(2,3,2,3,_u_a);
  scilab_rt_display_s0d2_("a",2,3,_u_a);
  double _u_b[2][2];
  _u_b[0][0]=1;
  _u_b[0][1]=2;
  _u_b[1][0]=3;
  _u_b[1][1]=5;
  scilab_rt_display_s0d2_("b",2,2,_u_b);
  double _tmpxx0[2][2];
  scilab_rt_cos_i2_d2(2,2,_u_b,2,2,_tmpxx0);
  
  scilab_rt_assign_d2_d2(2,2,_tmpxx0,2,2,_u_b);
  scilab_rt_display_s0d2_("b",2,2,_u_b);

  scilab_rt_terminate();
}

