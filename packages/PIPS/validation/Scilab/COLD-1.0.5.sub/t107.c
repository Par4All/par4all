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

  /*  t107.sce: testing fix function */
  double _u_a[2][3];
  _u_a[0][0]=1.5;
  _u_a[0][1]=6.8;
  _u_a[0][2]=(-9.1);
  _u_a[1][0]=8.2;
  _u_a[1][1]=65.5;
  _u_a[1][2]=(-78.123);
  scilab_rt_display_s0d2_("a",2,3,_u_a);
  int _u_b[2][3];
  scilab_rt_fix_d2_i2(2,3,_u_a,2,3,_u_b);
  scilab_rt_display_s0i2_("b",2,3,_u_b);
  int _u_c = scilab_rt_fix_d0_(_u_a[0][0]);
  scilab_rt_display_s0i0_("c",_u_c);

  scilab_rt_terminate();
}

