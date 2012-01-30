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

  /*  t108.sce _ linspace function */
  double _u_a[1][10];
  scilab_rt_linspace_d0d0i0_d2(1,6,10,1,10,_u_a);
  scilab_rt_display_s0d2_("a",1,10,_u_a);
  double _u_b[1][10];
  scilab_rt_linspace_d0d0i0_d2((-1),6,10,1,10,_u_b);
  scilab_rt_display_s0d2_("b",1,10,_u_b);
  double _u_c[1][10];
  scilab_rt_linspace_d0d0i0_d2(1,(-6),10,1,10,_u_c);
  scilab_rt_display_s0d2_("c",1,10,_u_c);
  double _u_d[1][10];
  scilab_rt_linspace_d0d0i0_d2((-1),(-6),10,1,10,_u_d);
  scilab_rt_display_s0d2_("d",1,10,_u_d);
  double _u_a1[1][10];
  scilab_rt_linspace_d0d0i0_d2(1.3,4.7,10,1,10,_u_a1);
  scilab_rt_display_s0d2_("a1",1,10,_u_a1);
  double _u_b1[1][10];
  scilab_rt_linspace_d0d0i0_d2((-1.3),4.7,10,1,10,_u_b1);
  scilab_rt_display_s0d2_("b1",1,10,_u_b1);
  double _u_c1[1][10];
  scilab_rt_linspace_d0d0i0_d2(1.3,(-4.7),10,1,10,_u_c1);
  scilab_rt_display_s0d2_("c1",1,10,_u_c1);
  double _u_d1[1][10];
  scilab_rt_linspace_d0d0i0_d2((-1.3),(-4.7),10,1,10,_u_d1);
  scilab_rt_display_s0d2_("d1",1,10,_u_d1);

  scilab_rt_terminate();
}

