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

  /*  t191.sce: p4a */
  double _u_a[10][10];
  scilab_rt_ones_i0i0_d2(10,10,10,10,_u_a);
  for (int _u_i=1; _u_i<=10; _u_i++) {
    for (int _u_j=1; _u_j<=10; _u_j++) {
      int _tmpxx0 = (_u_i+_u_j);
      _u_a[_u_i-1][_u_j-1] = _tmpxx0;
    }
  }
  double _u_b;
  scilab_rt_max_d2_d0(10,10,_u_a,&_u_b);
  scilab_rt_display_s0d0_("b",_u_b);

  scilab_rt_terminate();
}

