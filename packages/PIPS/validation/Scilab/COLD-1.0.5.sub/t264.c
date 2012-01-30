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

  /*  t264.sce _ vector[1,3]*vector[3,1]=scalar */
  double _u_a[1][3];
  scilab_rt_ones_i0i0_d2(1,3,1,3,_u_a);
  double _u_b[3][1];
  scilab_rt_ones_i0i0_d2(3,1,3,1,_u_b);
  double _u_c;
  scilab_rt_mul_d2d2_d0(1,3,_u_a,3,1,_u_b,&_u_c);
  scilab_rt_display_s0d0_("c",_u_c);
  double _u_d[3][3];
  scilab_rt_mul_d2d2_d2(3,1,_u_b,1,3,_u_a,3,3,_u_d);
  scilab_rt_display_s0d2_("d",3,3,_u_d);
  double _tmpxx0[1][3];
  scilab_rt_ones_i0i0_d2(1,3,1,3,_tmpxx0);
  double complex _u_ac[1][3];
  scilab_rt_mul_d2z0_z2(1,3,_tmpxx0,I,1,3,_u_ac);
  scilab_rt_display_s0z2_("ac",1,3,_u_ac);
  double _tmpxx1[3][1];
  scilab_rt_ones_i0i0_d2(3,1,3,1,_tmpxx1);
  double complex _u_bc[3][1];
  scilab_rt_mul_d2z0_z2(3,1,_tmpxx1,I,3,1,_u_bc);
  double complex _u_cc;
  scilab_rt_mul_z2z2_z0(1,3,_u_ac,3,1,_u_bc,&_u_cc);
  scilab_rt_display_s0z0_("cc",_u_cc);
  double complex _u_dc[3][3];
  scilab_rt_mul_z2z2_z2(3,1,_u_bc,1,3,_u_ac,3,3,_u_dc);
  scilab_rt_display_s0z2_("dc",3,3,_u_dc);

  scilab_rt_terminate();
}

