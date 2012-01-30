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
  int _u_b[1][10];
  scilab_rt_and_d2i0_i2(10,10,_u_a,1,1,10,_u_b);
  scilab_rt_display_s0i2_("b",1,10,_u_b);
  int _u_c[10][1];
  scilab_rt_and_d2i0_i2(10,10,_u_a,2,10,1,_u_c);
  scilab_rt_display_s0i2_("c",10,1,_u_c);

  scilab_rt_terminate();
}

