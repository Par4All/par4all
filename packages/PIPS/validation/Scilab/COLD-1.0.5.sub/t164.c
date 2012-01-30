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

  /*  t164.sce - aref with triplet (boundaries unknown) */
  int _u_N1 = 5;
  scilab_rt_display_s0i0_("N1",_u_N1);
  int _u_N2 = 5;
  scilab_rt_display_s0i0_("N2",_u_N2);
  int _u_N = (_u_N1+_u_N2);
  scilab_rt_display_s0i0_("N",_u_N);
  double _u_dx[10][1];
  scilab_rt_zeros_i0i0_d2(_u_N,1,10,1,_u_dx);

  for(int i=5; i<10; i++) {
    _u_dx[i][0] = 1;
  }
  scilab_rt_display_s0d2_("dx",10,1,_u_dx);

  scilab_rt_terminate();
}

