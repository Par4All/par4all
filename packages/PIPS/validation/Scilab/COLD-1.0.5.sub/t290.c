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

  /*  t290.sce _ pragma */
  //scilab pragma: a double complex 
  double _tmpCT0[1][3];
  scilab_rt_ones_i0i0_d2(1,3,1,3,_tmpCT0);
  double complex _u_a[1][3];
  scilab_rt_assign_d2_z2(1,3,_tmpCT0,1,3,_u_a);
  scilab_rt_display_s0z2_("a",1,3,_u_a);
  double complex _tmpxx0 = (2+(3*I));
  _u_a[0][(1-1)] = _tmpxx0;
  scilab_rt_display_s0z2_("a",1,3,_u_a);

  scilab_rt_terminate();
}

