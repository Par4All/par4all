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

  /*  t258.sce: ones, zeros, rand with 3D matrices */
  double _u_a[3][3][3];
  scilab_rt_ones_i0i0i0_d3(3,3,3,3,3,3,_u_a);
  double _u_t1[3][3][3];
  scilab_rt_ones_d3_d3(3,3,3,_u_a,3,3,3,_u_t1);
  scilab_rt_display_s0d3_("t1",3,3,3,_u_t1);
  double _u_t2[3][3][3];
  scilab_rt_zeros_d3_d3(3,3,3,_u_a,3,3,3,_u_t2);
  scilab_rt_display_s0d3_("t2",3,3,3,_u_t2);
  double _u_t3[3][3][3];
  scilab_rt_rand_d3_d3(3,3,3,_u_a,3,3,3,_u_t3);
  scilab_rt_display_s0d3_("t3",3,3,3,_u_t3);

  scilab_rt_terminate();
}

