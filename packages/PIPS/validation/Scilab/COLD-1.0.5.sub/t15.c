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
  double _u_b = _u_a[0][0];
  scilab_rt_display_s0d0_("b",_u_b);

  scilab_rt_terminate();
}

