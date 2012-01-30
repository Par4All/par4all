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

  double _u_a[3][5];
  scilab_rt_ones_i0i0_d2(3,5,3,5,_u_a);
  int _u_r;
  scilab_rt_size_d2s0_i0(3,5,_u_a,"r",&_u_r);
  scilab_rt_display_s0i0_("r",_u_r);
  int _u_c;
  scilab_rt_size_d2s0_i0(3,5,_u_a,"c",&_u_c);
  scilab_rt_display_s0i0_("c",_u_c);
  for (int _u_i=1; _u_i<=_u_r; _u_i++) {
    for (int _u_j=1; _u_j<=_u_c; _u_j++) {
      int _tmpxx0 = (_u_j+_u_i);
      _u_a[_u_i-1][_u_j-1] = _tmpxx0;
    }
  }
  scilab_rt_display_s0d2_("a",3,5,_u_a);

  scilab_rt_terminate();
}

