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

  /*  constant propagation */
  int _u_N = 10;
  double _u_a[10][11];
  scilab_rt_ones_i0i0_d2(_u_N,(_u_N+1),10,11,_u_a);
  scilab_rt_display_s0d2_("a",10,11,_u_a);

  scilab_rt_terminate();
}

