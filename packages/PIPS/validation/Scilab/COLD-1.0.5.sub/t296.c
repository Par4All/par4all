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

  /*  t296.sce _ Fixed a bug when the RT is called with the new type of an array */
  double _u_a[1][3];
  scilab_rt_ones_i0i0_d2(1,3,1,3,_u_a);
  scilab_rt_display_s0d2_("a",1,3,_u_a);
  int _tmpCT0[1][3];
  scilab_rt_floor_d2_i2(1,3,_u_a,1,3,_tmpCT0);
  double _u_b[1][3];
  scilab_rt_assign_i2_d2(1,3,_tmpCT0,1,3,_u_b);
  scilab_rt_display_s0d2_("b",1,3,_u_b);
  
  scilab_rt_ones_i0i0_d2(1,3,1,3,_u_b);
  scilab_rt_display_s0d2_("b",1,3,_u_b);
  _u_b[0][(1-1)] = 1.42;
  scilab_rt_display_s0d2_("b",1,3,_u_b);

  scilab_rt_terminate();
}

